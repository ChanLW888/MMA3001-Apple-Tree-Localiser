#!/usr/bin/env python3
"""Run qualitative winter RGB-D inference and model-only timing.

Reconstructs training MLP/basic/improved U-Nets without ImageNet weight retrieval, then
strictly loads trusted local checkpoints. Uses training-compatible preprocessing. No
winter GT is required; predictions cannot establish winter accuracy. Timing excludes
preprocessing, transfer, sigmoid and saving.

Requires winter RGB-D pairs and all three local model checkpoints. Writes masks,
overlays, panels, per_image_timings.csv, summary.csv and benchmark_metadata.json
under outputs/winter_inference_benchmark by default. Reruns overwrite matching names.

Example:
    See README.md for the invocation supplying all three checkpoints.
"""
import argparse, csv, json, re, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights

RGB_RE = re.compile(r'_RGB-', re.I)
DEPTH_RE = re.compile(r'_D-', re.I)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

def crop_square(im):
    """Crop a PIL image to its largest centred square.

    Args:
        im: PIL.Image.Image to crop.

    Returns:
        PIL.Image.Image: Square with side min(width, height).
    """
    w,h=im.size; s=min(w,h); l=(w-s)//2; t=(h-s)//2
    return im.crop((l,t,l+s,t+s))

def sample_key(p, pat):
    """Replace the first modality token, retaining frame identity.

    Args:
        p: pathlib.Path to a winter image.
        pat: re.Pattern for RGB or depth token.

    Returns:
        str: Stem with first token replaced by _TIME-.
    """
    return pat.sub('_TIME-', p.stem, count=1)

def discover(root):
    """Find complete winter RGB/depth pairs without requiring GT.

    Reports unmatched RGB images and excludes incomplete pairs. Filenames do not verify
    geometric alignment.

    Args:
        root: pathlib.Path; recursively searched dataset root.

    Returns:
        list[dict]: Sorted key, rgb and depth records.

    Raises:
        RuntimeError: If duplicate keys occur or no pairs exist.
    """
    rgbs=sorted(p for p in root.rglob('*.png') if RGB_RE.search(p.stem) and 'winter' in p.stem.lower())
    depths=sorted(p for p in root.rglob('*.png') if DEPTH_RE.search(p.stem) and 'winter' in p.stem.lower())
    rm={sample_key(p,RGB_RE):p for p in rgbs}; dm={sample_key(p,DEPTH_RE):p for p in depths}
    if len(rm)!=len(rgbs) or len(dm)!=len(depths): raise RuntimeError('Duplicate winter RGB/depth sample keys found.')
    pairs=[{'key':k,'rgb':rm[k],'depth':dm[k]} for k in sorted(rm.keys() & dm.keys())]
    print(f'Winter RGB: {len(rgbs)} | depth: {len(depths)} | matched pairs: {len(pairs)}')
    if not pairs: raise RuntimeError(f'No matched winter RGB-D pairs under {root}. Check filenames and --data-root.')
    if len(pairs)<len(rgbs): print(f'WARNING: {len(rgbs)-len(pairs)} RGB images had no matching depth image.')
    return pairs

def preprocess(s,size):
    """Apply training-compatible RGB-D preprocessing.

    Centre-crops both modalities; RGB is bilinear, /255 then ImageNet standardised.
    Depth is nearest neighbour, grayscale /255, not metric distance.

    Args:
        s: dict with rgb and depth pathlib.Path entries.
        size: int; positive square input side in pixels, normally 256.

    Returns:
        tuple[torch.Tensor, PIL.Image.Image, PIL.Image.Image]: Float32 [4, size,
            size] input, resized RGB and L-mode depth.

    Raises:
        OSError: If images cannot be decoded.
        ValueError: If resize dimensions are invalid.
    """
    rgbim=crop_square(Image.open(s['rgb']).convert('RGB')).resize((size,size),Image.Resampling.BILINEAR)
    rgb=np.asarray(rgbim,dtype=np.float32)/255.
    rgbn=(rgb-MEAN)/STD
    dim=crop_square(Image.open(s['depth']).convert('L')).resize((size,size),Image.Resampling.NEAREST)
    d=np.asarray(dim,dtype=np.float32)/255.
    x=np.concatenate([rgbn.transpose(2,0,1),d[None]],axis=0).astype(np.float32)
    return torch.from_numpy(x),rgbim,dim

class PixelMLP(nn.Module):
    """Reconstruct the training pixel MLP for checkpoint inference."""
    def __init__(self,hidden_dims=(64,32,16)):
        """Construct four-feature pixel layers with matching checkpoint names.

        Args:
            hidden_dims: tuple[int, ...]; positive hidden widths matching the
                checkpoint.

        Returns:
            None.
        """
        super().__init__(); layers=[]; prev=4
        for h in hidden_dims: layers += [nn.Linear(prev,h),nn.ReLU(inplace=True)]; prev=h
        layers.append(nn.Linear(prev,1)); self.network=nn.Sequential(*layers)
    def forward(self,x):
        """Classify each pixel independently.

        Args:
            x: torch.Tensor; [B, 4, H, W] preprocessed RGB-D.

        Returns:
            torch.Tensor: Raw [B, 1, H, W] branch logits.

        Raises:
            RuntimeError: If feature count or dtype/device is incompatible.
        """
        b,c,h,w=x.shape; p=x.permute(0,2,3,1).reshape(-1,c)
        return self.network(p).reshape(b,h,w).unsqueeze(1)

def decoder_block(in_ch,mid_ch,out_ch):
    """Construct a two-convolution decoder feature block.

    Args:
        in_ch: int; positive input channels.
        mid_ch: int; positive hidden channels.
        out_ch: int; positive output channels.

    Returns:
        torch.nn.Sequential: 3x3 convolution, batch-normalisation and ReLU layers
            preserving spatial size.
    """
    return nn.Sequential(nn.Conv2d(in_ch,mid_ch,3,padding=1,bias=False),nn.BatchNorm2d(mid_ch),nn.ReLU(inplace=True),nn.Conv2d(mid_ch,out_ch,3,padding=1,bias=False),nn.BatchNorm2d(out_ch),nn.ReLU(inplace=True))
class BasicUNet(nn.Module):
    """Predict branch logits with a four-channel ResNet-34 U-Net.

    Encoder skip features restore spatial detail during decoding. Layers match training
    checkpoint keys; construction uses weights=None.
    """
    def __init__(self):
        """Construct encoder and decoder layers.

        Returns:
            None.
        """
        super().__init__(); b=resnet34(weights=None); old=b.conv1
        b.conv1=nn.Conv2d(4,old.out_channels,old.kernel_size,old.stride,old.padding,bias=False)
        self.stem=nn.Sequential(b.conv1,b.bn1,b.relu); self.maxpool=b.maxpool
        self.enc1=b.layer1; self.enc2=b.layer2; self.enc3=b.layer3; self.enc4=b.layer4
        self.up4=decoder_block(512,256,256); self.up3=decoder_block(512,128,128)
        self.up2=decoder_block(256,64,64); self.up1=decoder_block(128,64,64); self.up0=decoder_block(128,32,32); self.out=nn.Conv2d(32,1,1)
    @staticmethod
    def up(x,ref):
        """Match a feature map to an encoder grid before concatenation.

        Args:
            x: torch.Tensor; floating [B, C, H, W] features.
            ref: torch.Tensor; last two dimensions define destination size.

        Returns:
            torch.Tensor: Bilinear [B, C, Href, Wref] features, align_corners=False.

        Raises:
            ValueError: If interpolation receives invalid spatial rank.
        """
        return F.interpolate(x,size=ref.shape[-2:],mode='bilinear',align_corners=False)
    def forward(self,x):
        """Decode RGB-D features into full-resolution branch logits.

        Args:
            x: torch.Tensor; floating [B, 4, H, W] batch using training
                preprocessing.

        Returns:
            torch.Tensor: Raw [B, 1, H, W] logits; sigmoid converts to
                probabilities.

        Raises:
            RuntimeError: If channels, spatial size or dtype/device is incompatible.
        """
        size=x.shape[-2:]; s0=self.stem(x); e1=self.enc1(self.maxpool(s0)); e2=self.enc2(e1); e3=self.enc3(e2); e4=self.enc4(e3)
        d4=self.up4(self.up(e4,e3)); d3=self.up3(torch.cat([d4,e3],1)); d2=self.up2(torch.cat([self.up(d3,e2),e2],1)); d1=self.up1(torch.cat([self.up(d2,e1),e1],1)); d0=self.up0(torch.cat([self.up(d1,s0),s0],1))
        return self.out(F.interpolate(d0,size=size,mode='bilinear',align_corners=False))

class ConvNormAct(nn.Module):
    """Extract features with convolution, batch normalisation and ReLU."""
    def __init__(self,ic,oc,k=3,d=1):
        """Build a padded feature convolution.

        Args:
            ic: int; positive input channels.
            oc: int; positive output channels.
            k: int; positive kernel side, normally odd.
            d: int; positive dilation rate.

        Returns:
            None.
        """
        super().__init__(); self.block=nn.Sequential(nn.Conv2d(ic,oc,k,padding=d*(k//2),dilation=d,bias=False),nn.BatchNorm2d(oc),nn.ReLU(inplace=True))
    def forward(self,x):
        """Transform a batch of features.

        Args:
            x: torch.Tensor; [B, input_channels, H, W].

        Returns:
            torch.Tensor: [B, output_channels, H, W] features for odd kernels; not
                branch logits.

        Raises:
            RuntimeError: If channels, batch-normalisation input or device is
                incompatible.
        """
        return self.block(x)
class ResidualDecoderBlock(nn.Module):
    """Fuse resized decoder and skip features through a residual path."""
    def __init__(self,ic,sc,oc):
        """Construct convolutions and a channel-matching shortcut.

        Args:
            ic: int; positive decoder channels.
            sc: int; skip channels, or 0 when no skip will be provided.
            oc: int; positive output channels.

        Returns:
            None.
        """
        super().__init__(); self.conv1=ConvNormAct(ic+sc,oc); self.conv2=nn.Sequential(nn.Conv2d(oc,oc,3,padding=1,bias=False),nn.BatchNorm2d(oc)); self.shortcut=nn.Identity() if ic+sc==oc else nn.Sequential(nn.Conv2d(ic+sc,oc,1,bias=False),nn.BatchNorm2d(oc)); self.relu=nn.ReLU(inplace=True)
    def forward(self,x,skip=None):
        """Resize to the skip grid, concatenate and add a residual projection.

        Args:
            x: torch.Tensor; [B, decoder_channels, H, W].
            skip: torch.Tensor or None; [B, skip_channels, Hs, Ws] encoder features.

        Returns:
            torch.Tensor: [B, output_channels, Hs, Ws] with skip; otherwise retains
                input grid.

        Raises:
            RuntimeError: If concatenated channels or devices are incompatible.
        """
        if skip is not None: x=torch.cat([F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False),skip],1)
        return self.relu(self.conv2(self.conv1(x))+self.shortcut(x))
class AttentionGate(nn.Module):
    """Weight encoder skip features using decoder context."""
    def __init__(self,gc,sc,ic):
        """Construct context projections and scalar spatial attention.

        Args:
            gc: int; positive context channels.
            sc: int; positive encoder channels.
            ic: int; positive intermediate channels.

        Returns:
            None.
        """
        super().__init__(); self.gate=nn.Sequential(nn.Conv2d(gc,ic,1,bias=False),nn.BatchNorm2d(ic)); self.skip=nn.Sequential(nn.Conv2d(sc,ic,1,bias=False),nn.BatchNorm2d(ic)); self.score=nn.Sequential(nn.ReLU(inplace=True),nn.Conv2d(ic,1,1),nn.Sigmoid())
    def forward(self,g,s):
        """Apply sigmoid attention while retaining the skip shape.

        Args:
            g: torch.Tensor; [B, gate_channels, Hg, Wg] context.
            s: torch.Tensor; [B, skip_channels, Hs, Ws] encoder features.

        Returns:
            torch.Tensor: Weighted skip [B, skip_channels, Hs, Ws], not a
                one-channel prediction.

        Raises:
            RuntimeError: If channels, batch dimensions or devices are incompatible.
        """
        g=F.interpolate(g,size=s.shape[-2:],mode='bilinear',align_corners=False); return s*self.score(self.gate(g)+self.skip(s))
class ASPP(nn.Module):
    """Combine parallel convolutions at dilation rates 1, 2, 4 and 6.

    The rate-1 branch uses a 1x1 kernel. Projection fuses concatenated features; dropout
    is active only in training mode.
    """
    def __init__(self,ic,oc):
        """Construct four branches and a context projection.

        Args:
            ic: int; positive input channels.
            oc: int; output channels, at least 4; branch width is
                floor(output_channels/4).

        Returns:
            None.
        """
        super().__init__(); bc=oc//4; self.branches=nn.ModuleList([ConvNormAct(ic,bc,1),ConvNormAct(ic,bc,d=2),ConvNormAct(ic,bc,d=4),ConvNormAct(ic,bc,d=6)]); self.project=nn.Sequential(nn.Conv2d(bc*4,oc,1,bias=False),nn.BatchNorm2d(oc),nn.ReLU(inplace=True),nn.Dropout2d(.10))
    def forward(self,x):
        """Fuse multiscale features without changing the spatial grid.

        Args:
            x: torch.Tensor; [B, input_channels, H, W].

        Returns:
            torch.Tensor: [B, output_channels, H, W] context features, not branch
                logits.

        Raises:
            RuntimeError: If convolution or batch-normalisation input is
                incompatible.
        """
        return self.project(torch.cat([b(x) for b in self.branches],1))
class ImprovedUNet(nn.Module):
    """Predict branch logits with a four-channel ResNet-34 U-Net.

    ASPP supplies multiscale context, attention weights skip features, and residual
    blocks fuse them. Layers match training checkpoint keys; construction uses
    weights=None.
    """
    def __init__(self):
        """Construct encoder and decoder layers.

        Returns:
            None.
        """
        super().__init__(); b=resnet34(weights=None); old=b.conv1; b.conv1=nn.Conv2d(4,old.out_channels,old.kernel_size,old.stride,old.padding,bias=False)
        self.stem=nn.Sequential(b.conv1,b.bn1,b.relu); self.maxpool=b.maxpool; self.enc1=b.layer1; self.enc2=b.layer2; self.enc3=b.layer3; self.enc4=b.layer4
        self.context=ASPP(512,256); self.att3=AttentionGate(256,256,128); self.att2=AttentionGate(192,128,64); self.att1=AttentionGate(128,64,32); self.att0=AttentionGate(96,64,32)
        self.dec3=ResidualDecoderBlock(256,256,192); self.dec2=ResidualDecoderBlock(192,128,128); self.dec1=ResidualDecoderBlock(128,64,96); self.dec0=ResidualDecoderBlock(96,64,64); self.refine=nn.Sequential(ConvNormAct(64,48),nn.Dropout2d(.05),ConvNormAct(48,32)); self.out=nn.Conv2d(32,1,1)
    def forward(self,x):
        """Decode RGB-D features into full-resolution branch logits.

        Args:
            x: torch.Tensor; floating [B, 4, H, W] batch using training
                preprocessing.

        Returns:
            torch.Tensor: Raw [B, 1, H, W] logits; sigmoid converts to
                probabilities.

        Raises:
            RuntimeError: If channels, spatial size or dtype/device is incompatible.
        """
        size=x.shape[-2:]; s0=self.stem(x); e1=self.enc1(self.maxpool(s0)); e2=self.enc2(e1); e3=self.enc3(e2); e4=self.enc4(e3); z=self.context(e4)
        d3=self.dec3(z,self.att3(z,e3)); d2=self.dec2(d3,self.att2(d3,e2)); d1=self.dec1(d2,self.att1(d2,e1)); d0=self.dec0(d1,self.att0(d1,s0)); d0=F.interpolate(d0,size=size,mode='bilinear',align_corners=False)
        return self.out(self.refine(d0))

def load_checkpoint(model,path,device):
    """Load matching parameters strictly and select evaluation mode.

    Loads on CPU with weights_only=False before transfer. Checkpoints must be trusted
    because pickle deserialisation can execute payloads.

    Args:
        model: torch.nn.Module with matching state keys.
        path: str or pathlib.Path to a trusted local PyTorch checkpoint.
        device: torch.device for the returned model.

    Returns:
        tuple[torch.nn.Module, dict]: Device-mapped eval model and original
            checkpoint; accepts model_state_dict or a bare state dictionary.

    Raises:
        OSError: If reading fails.
        RuntimeError: If strict loading or device transfer fails.
    """
    ckpt=torch.load(path,map_location='cpu',weights_only=False); state=ckpt.get('model_state_dict',ckpt); model.load_state_dict(state,strict=True); return model.to(device).eval(),ckpt

def save_outputs(rgbim,depthim,mask,outdir,stem):
    """Save a prediction, orange overlay and four-panel comparison.

    Suffixes are _pred_mask.png, _overlay.png and _comparison.png. No GT or quantitative
    segmentation accuracy is produced.

    Args:
        rgbim: PIL.Image.Image; preprocessed RGB.
        depthim: PIL.Image.Image; same-size L-mode depth.
        mask: PIL.Image.Image; same-size L-mode binary prediction, 0/255.
        outdir: pathlib.Path; created if needed.
        stem: str; filesystem-safe basename without extension.

    Returns:
        None.

    Raises:
        OSError: If directory creation or PNG writing fails.
        ValueError: If arrays have incompatible dimensions.
    """
    outdir.mkdir(parents=True,exist_ok=True); mask.save(outdir/f'{stem}_pred_mask.png')
    rgb=np.asarray(rgbim.convert('RGB')); m=np.asarray(mask)>0; over=rgb.copy(); orange=np.array([255,140,0],np.uint8); over[m]=(0.45*over[m]+0.55*orange).astype(np.uint8); Image.fromarray(over).save(outdir/f'{stem}_overlay.png')
    dep=np.repeat(np.asarray(depthim,dtype=np.uint8)[...,None],3,2); pred=np.repeat(np.asarray(mask)[...,None],3,2); panels=[rgb,dep,pred,over]; labels=['RGB','Depth','Predicted mask','Overlay']; w,h=rgbim.size; canvas=Image.new('RGB',(w*4,h+32),'white'); draw=ImageDraw.Draw(canvas)
    for i,(p,label) in enumerate(zip(panels,labels)): canvas.paste(Image.fromarray(p),(i*w,32)); draw.text((i*w+8,9),label,fill='black')
    canvas.save(outdir/f'{stem}_comparison.png')

def sync(device):
    """Finish CUDA kernels before wall-clock timing.

    Args:
        device: torch.device; CPU does nothing, CUDA synchronises that device.

    Returns:
        None.

    Raises:
        RuntimeError: If CUDA synchronisation fails.
    """
    if device.type=='cuda': torch.cuda.synchronize(device)

def benchmark(model,samples,name,outroot,device,size,threshold,warmup):
    """Measure model-only latency and export winter predictions.

    Caller sets eval mode; inference_mode disables gradients. Timing excludes
    preprocessing, transfer, sigmoid, thresholding and saving. Sample standard deviation
    uses ddof=1 for at least two images; absent means are None.

    Args:
        model: torch.nn.Module; eval model on device mapping RGB-D to branch logits.
        samples: list[dict] from discover.
        name: str; model label and output subdirectory.
        outroot: pathlib.Path; root output directory.
        device: torch.device matching model parameters.
        size: int; positive input side matching training geometry.
        threshold: float in [0, 1]; inclusive sigmoid cutoff.
        warmup: int; nonnegative untimed first-sample forward passes.

    Returns:
        tuple[list[dict], dict]: Per-image timings and summary in milliseconds;
            throughput is model-only images/second.

    Raises:
        OSError: If input reading or prediction saving fails.
        RuntimeError: If inference or synchronisation fails.
    """
    outdir=outroot/name; outdir.mkdir(parents=True,exist_ok=True); rows=[]; times=[]
    with torch.inference_mode():
        if samples:
            x0,_,_=preprocess(samples[0],size); x0=x0[None].to(device)
            for _ in range(warmup): model(x0)
            sync(device)
        for i,s in enumerate(samples,1):
            x,rgbim,depthim=preprocess(s,size); xd=x[None].to(device); sync(device); t=time.perf_counter(); logits=model(xd); sync(device); ms=(time.perf_counter()-t)*1000; times.append(ms)
            probs=torch.sigmoid(logits)[0,0].cpu().numpy(); mask=Image.fromarray((probs>=threshold).astype(np.uint8)*255,mode='L'); stem=re.sub(r'[^A-Za-z0-9_.-]+','_',s['key']); save_outputs(rgbim,depthim,mask,outdir,stem)
            rows.append({'model':name,'image_key':s['key'],'rgb_path':str(s['rgb']),'depth_path':str(s['depth']),'inference_time_ms':f'{ms:.4f}','input_size':f'{size}x{size}','threshold':threshold,'device':str(device)})
            print(f'[{name}] {i}/{len(samples)} {s["key"]}: {ms:.3f} ms')
    a=np.asarray(times,dtype=float)
    summary={'model':name,'images_processed':len(times),'mean_inference_ms_per_image':float(a.mean()) if len(a) else None,'median_inference_ms_per_image':float(np.median(a)) if len(a) else None,'std_inference_ms':float(a.std(ddof=1)) if len(a)>1 else 0.0,'min_inference_ms':float(a.min()) if len(a) else None,'max_inference_ms':float(a.max()) if len(a) else None,'images_per_second_model_only':float(1000/a.mean()) if len(a) and a.mean()>0 else None,'output_directory':str(outdir),'device':str(device),'timed_region':'model forward pass only; excludes preprocessing, transfer, sigmoid, and disk writes'}
    return rows,summary

def main():
    """Load three checkpoints and benchmark paired winter images.

    Requires --data-root and --mlp-checkpoint, --unet-checkpoint, --improved-checkpoint.
    Saves per_image_timings.csv, summary.csv, benchmark_metadata.json and model-specific
    images. Input size comes from CLI, not checkpoint metadata. Requires no GT.

    Returns:
        None.

    Raises:
        SystemExit: If required CLI arguments are missing or --help is requested.
        FileNotFoundError: If a required path is absent.
        RuntimeError: If pairing, loading or inference fails.
        OSError: If inputs/outputs cannot be accessed.
    """
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--data-root',type=Path,required=True); ap.add_argument('--mlp-checkpoint',type=Path,required=True); ap.add_argument('--unet-checkpoint',type=Path,required=True); ap.add_argument('--improved-checkpoint',type=Path,required=True); ap.add_argument('--output-dir',type=Path,default=Path('outputs/winter_inference_benchmark')); ap.add_argument('--input-size',type=int,default=256); ap.add_argument('--threshold',type=float,default=.5); ap.add_argument('--warmup',type=int,default=5); args=ap.parse_args()
    for p in [args.data_root,args.mlp_checkpoint,args.unet_checkpoint,args.improved_checkpoint]:
        if not p.exists(): raise FileNotFoundError(f'Path does not exist: {p}')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('Device:',device)
    if device.type=='cuda': print('GPU:',torch.cuda.get_device_name(0))
    samples=discover(args.data_root); args.output_dir.mkdir(parents=True,exist_ok=True)
    ckpt=torch.load(args.mlp_checkpoint,map_location='cpu',weights_only=False); mlp=PixelMLP(tuple(ckpt.get('hidden_dims',[64,32,16]))); mlp,_=load_checkpoint(mlp,args.mlp_checkpoint,device)
    unet,_=load_checkpoint(BasicUNet(),args.unet_checkpoint,device); improved,_=load_checkpoint(ImprovedUNet(),args.improved_checkpoint,device)
    allrows=[]; summaries=[]
    for model,name in [(mlp,'pixel_mlp'),(unet,'basic_unet'),(improved,'improved_unet')]:
        print(f'\n=== Benchmarking {name} ==='); rows,summ=benchmark(model,samples,name,args.output_dir,device,args.input_size,args.threshold,args.warmup); allrows+=rows; summaries.append(summ)
        del model
        if device.type=='cuda': torch.cuda.empty_cache()
    with (args.output_dir/'per_image_timings.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['model','image_key','rgb_path','depth_path','inference_time_ms','input_size','threshold','device']); w.writeheader(); w.writerows(allrows)
    with (args.output_dir/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0].keys())); w.writeheader(); w.writerows(summaries)
    meta={'data_root':str(args.data_root),'mlp_checkpoint':str(args.mlp_checkpoint),'basic_unet_checkpoint':str(args.unet_checkpoint),'improved_unet_checkpoint':str(args.improved_checkpoint),'images':len(samples),'input_size':args.input_size,'threshold':args.threshold,'warmup_iterations_per_model':args.warmup,'device':str(device),'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None,'notes':['Winter images used for qualitative inference only; no winter ground-truth masks required.','Timing is forward-pass only; preprocessing, transfer, thresholding and saving are excluded.','Outputs are saved under separate model-specific directories.']}
    (args.output_dir/'benchmark_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('\n=== SUMMARY ===')
    for s in summaries: print(f"{s['model']}: {s['images_processed']} images, mean {s['mean_inference_ms_per_image']:.3f} ms/image, median {s['median_inference_ms_per_image']:.3f} ms/image")
    print('Per-image timings:',(args.output_dir/'per_image_timings.csv').resolve()); print('Summary:',(args.output_dir/'summary.csv').resolve()); print('Outputs:',args.output_dir.resolve())
if __name__=='__main__': main()

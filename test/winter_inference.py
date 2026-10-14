#!/usr/bin/env python3
"""Run all winter RGB-D pairs through the MLP, basic U-Net and improved U-Net.

Saves masks/overlays/comparison panels per model plus per-image and summary timing CSVs.
Timing is model forward-pass latency only (not preprocessing, transfer, or saving).
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
    w,h=im.size; s=min(w,h); l=(w-s)//2; t=(h-s)//2
    return im.crop((l,t,l+s,t+s))

def sample_key(p, pat): return pat.sub('_TIME-', p.stem, count=1)

def discover(root):
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
    rgbim=crop_square(Image.open(s['rgb']).convert('RGB')).resize((size,size),Image.Resampling.BILINEAR)
    rgb=np.asarray(rgbim,dtype=np.float32)/255.
    rgbn=(rgb-MEAN)/STD
    dim=crop_square(Image.open(s['depth']).convert('L')).resize((size,size),Image.Resampling.NEAREST)
    d=np.asarray(dim,dtype=np.float32)/255.
    x=np.concatenate([rgbn.transpose(2,0,1),d[None]],axis=0).astype(np.float32)
    return torch.from_numpy(x),rgbim,dim

class PixelMLP(nn.Module):
    def __init__(self,hidden_dims=(64,32,16)):
        super().__init__(); layers=[]; prev=4
        for h in hidden_dims: layers += [nn.Linear(prev,h),nn.ReLU(inplace=True)]; prev=h
        layers.append(nn.Linear(prev,1)); self.network=nn.Sequential(*layers)
    def forward(self,x):
        b,c,h,w=x.shape; p=x.permute(0,2,3,1).reshape(-1,c)
        return self.network(p).reshape(b,h,w).unsqueeze(1)

def decoder_block(in_ch,mid_ch,out_ch):
    return nn.Sequential(nn.Conv2d(in_ch,mid_ch,3,padding=1,bias=False),nn.BatchNorm2d(mid_ch),nn.ReLU(inplace=True),nn.Conv2d(mid_ch,out_ch,3,padding=1,bias=False),nn.BatchNorm2d(out_ch),nn.ReLU(inplace=True))
class BasicUNet(nn.Module):
    def __init__(self):
        super().__init__(); b=resnet34(weights=None); old=b.conv1
        b.conv1=nn.Conv2d(4,old.out_channels,old.kernel_size,old.stride,old.padding,bias=False)
        self.stem=nn.Sequential(b.conv1,b.bn1,b.relu); self.maxpool=b.maxpool
        self.enc1=b.layer1; self.enc2=b.layer2; self.enc3=b.layer3; self.enc4=b.layer4
        self.up4=decoder_block(512,256,256); self.up3=decoder_block(512,128,128)
        self.up2=decoder_block(256,64,64); self.up1=decoder_block(128,64,64); self.up0=decoder_block(128,32,32); self.out=nn.Conv2d(32,1,1)
    @staticmethod
    def up(x,ref): return F.interpolate(x,size=ref.shape[-2:],mode='bilinear',align_corners=False)
    def forward(self,x):
        size=x.shape[-2:]; s0=self.stem(x); e1=self.enc1(self.maxpool(s0)); e2=self.enc2(e1); e3=self.enc3(e2); e4=self.enc4(e3)
        d4=self.up4(self.up(e4,e3)); d3=self.up3(torch.cat([d4,e3],1)); d2=self.up2(torch.cat([self.up(d3,e2),e2],1)); d1=self.up1(torch.cat([self.up(d2,e1),e1],1)); d0=self.up0(torch.cat([self.up(d1,s0),s0],1))
        return self.out(F.interpolate(d0,size=size,mode='bilinear',align_corners=False))

class ConvNormAct(nn.Module):
    def __init__(self,ic,oc,k=3,d=1):
        super().__init__(); self.block=nn.Sequential(nn.Conv2d(ic,oc,k,padding=d*(k//2),dilation=d,bias=False),nn.BatchNorm2d(oc),nn.ReLU(inplace=True))
    def forward(self,x): return self.block(x)
class ResidualDecoderBlock(nn.Module):
    def __init__(self,ic,sc,oc):
        super().__init__(); self.conv1=ConvNormAct(ic+sc,oc); self.conv2=nn.Sequential(nn.Conv2d(oc,oc,3,padding=1,bias=False),nn.BatchNorm2d(oc)); self.shortcut=nn.Identity() if ic+sc==oc else nn.Sequential(nn.Conv2d(ic+sc,oc,1,bias=False),nn.BatchNorm2d(oc)); self.relu=nn.ReLU(inplace=True)
    def forward(self,x,skip=None):
        if skip is not None: x=torch.cat([F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False),skip],1)
        return self.relu(self.conv2(self.conv1(x))+self.shortcut(x))
class AttentionGate(nn.Module):
    def __init__(self,gc,sc,ic):
        super().__init__(); self.gate=nn.Sequential(nn.Conv2d(gc,ic,1,bias=False),nn.BatchNorm2d(ic)); self.skip=nn.Sequential(nn.Conv2d(sc,ic,1,bias=False),nn.BatchNorm2d(ic)); self.score=nn.Sequential(nn.ReLU(inplace=True),nn.Conv2d(ic,1,1),nn.Sigmoid())
    def forward(self,g,s):
        g=F.interpolate(g,size=s.shape[-2:],mode='bilinear',align_corners=False); return s*self.score(self.gate(g)+self.skip(s))
class ASPP(nn.Module):
    def __init__(self,ic,oc):
        super().__init__(); bc=oc//4; self.branches=nn.ModuleList([ConvNormAct(ic,bc,1),ConvNormAct(ic,bc,d=2),ConvNormAct(ic,bc,d=4),ConvNormAct(ic,bc,d=6)]); self.project=nn.Sequential(nn.Conv2d(bc*4,oc,1,bias=False),nn.BatchNorm2d(oc),nn.ReLU(inplace=True),nn.Dropout2d(.10))
    def forward(self,x): return self.project(torch.cat([b(x) for b in self.branches],1))
class ImprovedUNet(nn.Module):
    def __init__(self):
        super().__init__(); b=resnet34(weights=None); old=b.conv1; b.conv1=nn.Conv2d(4,old.out_channels,old.kernel_size,old.stride,old.padding,bias=False)
        self.stem=nn.Sequential(b.conv1,b.bn1,b.relu); self.maxpool=b.maxpool; self.enc1=b.layer1; self.enc2=b.layer2; self.enc3=b.layer3; self.enc4=b.layer4
        self.context=ASPP(512,256); self.att3=AttentionGate(256,256,128); self.att2=AttentionGate(192,128,64); self.att1=AttentionGate(128,64,32); self.att0=AttentionGate(96,64,32)
        self.dec3=ResidualDecoderBlock(256,256,192); self.dec2=ResidualDecoderBlock(192,128,128); self.dec1=ResidualDecoderBlock(128,64,96); self.dec0=ResidualDecoderBlock(96,64,64); self.refine=nn.Sequential(ConvNormAct(64,48),nn.Dropout2d(.05),ConvNormAct(48,32)); self.out=nn.Conv2d(32,1,1)
    def forward(self,x):
        size=x.shape[-2:]; s0=self.stem(x); e1=self.enc1(self.maxpool(s0)); e2=self.enc2(e1); e3=self.enc3(e2); e4=self.enc4(e3); z=self.context(e4)
        d3=self.dec3(z,self.att3(z,e3)); d2=self.dec2(d3,self.att2(d3,e2)); d1=self.dec1(d2,self.att1(d2,e1)); d0=self.dec0(d1,self.att0(d1,s0)); d0=F.interpolate(d0,size=size,mode='bilinear',align_corners=False)
        return self.out(self.refine(d0))

def load_checkpoint(model,path,device):
    ckpt=torch.load(path,map_location='cpu',weights_only=False); state=ckpt.get('model_state_dict',ckpt); model.load_state_dict(state,strict=True); return model.to(device).eval(),ckpt

def save_outputs(rgbim,depthim,mask,outdir,stem):
    outdir.mkdir(parents=True,exist_ok=True); mask.save(outdir/f'{stem}_pred_mask.png')
    rgb=np.asarray(rgbim.convert('RGB')); m=np.asarray(mask)>0; over=rgb.copy(); orange=np.array([255,140,0],np.uint8); over[m]=(0.45*over[m]+0.55*orange).astype(np.uint8); Image.fromarray(over).save(outdir/f'{stem}_overlay.png')
    dep=np.repeat(np.asarray(depthim,dtype=np.uint8)[...,None],3,2); pred=np.repeat(np.asarray(mask)[...,None],3,2); panels=[rgb,dep,pred,over]; labels=['RGB','Depth','Predicted mask','Overlay']; w,h=rgbim.size; canvas=Image.new('RGB',(w*4,h+32),'white'); draw=ImageDraw.Draw(canvas)
    for i,(p,label) in enumerate(zip(panels,labels)): canvas.paste(Image.fromarray(p),(i*w,32)); draw.text((i*w+8,9),label,fill='black')
    canvas.save(outdir/f'{stem}_comparison.png')

def sync(device):
    if device.type=='cuda': torch.cuda.synchronize(device)

def benchmark(model,samples,name,outroot,device,size,threshold,warmup):
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

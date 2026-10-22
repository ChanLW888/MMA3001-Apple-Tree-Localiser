dataset_organiser
=================

.. py:module:: dataset_organiser

Source: ``src/dataset_organiser.py``.

Statically extracted signatures and existing docstrings; no module execution.

Copy flat orchard PNGs into season/modality directories.

Running or importing immediately creates directories, copies matching files from
data/Dataset (With Summer GT) to data/Organised and prints counts/basic PNG checks.
Actual path constants use Windows separators and are relative to the working directory.
Copies can overwrite existing destinations. Does not resample or align images.

:raises OSError: If source listing, directory creation or copying fails.

.. rubric:: Example

From repository root: python src/dataset_organiser.py


.. py:function:: read_png(filename)

   Decode basic 8-bit grayscale or RGB PNG scanlines.
   
   Reverses PNG filters modulo 256, including Paeth. Does not validate signature, CRC
   or interlace method; this is a minimal checker, not a general PNG validator.
   
   :param filename: str or os.PathLike; non-interlaced PNG, colour type 0 or 2.
   
   :returns:
   
             Width, height, channels (1 or 3) and flat
                 row-major interleaved bytes in [0, 255].
   :rtype: tuple[int, int, int, list[int]]
   
   :raises ValueError: If bit depth, colour type or row filter is unsupported.
   :raises OSError: If reading fails.
   :raises zlib.error: If compressed data is invalid.
   :raises struct.error: If chunk fields are truncated.
   :raises IndexError: If decompressed rows are truncated.


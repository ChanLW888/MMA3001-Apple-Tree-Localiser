import os
import re
import shutil
import struct
import zlib


# Dataset locations
SOURCE_DIR = r"data\Dataset (With Summer GT)"
OUTPUT_DIR = r"data\Organised"

seasons = ["Winter", "Summer"]
image_types = ["RGB", "D", "GT"]


# Create the folder structure for the organised dataset
for season in seasons:
    for image_type in image_types:
        os.makedirs(
            os.path.join(OUTPUT_DIR, season, image_type),
            exist_ok=True
        )


# Match filenames such as:
# R01N01_Summer_RGB-12-40-50.png
pattern = re.compile(
    r"^(?P<tree>[^\_]+)"
    r"_(?P<season>Winter|Summer)"
    r"_(?P<type>RGB|D|GT)"
    r"-(?P<time>.+)\.png$",
    re.IGNORECASE
)

processed = 0
skipped = 0

print("\nSorting dataset...\n")

for filename in os.listdir(SOURCE_DIR):

    source_file = os.path.join(SOURCE_DIR, filename)

    if not os.path.isfile(source_file):
        continue

    match = pattern.match(filename)

    if not match:
        print("Skipped:", filename)
        skipped += 1
        continue

    season = match.group("season").capitalize()
    image_type = match.group("type").upper()

    destination = os.path.join(
        OUTPUT_DIR,
        season,
        image_type,
        filename
    )

    shutil.copy2(source_file, destination)
    processed += 1


# Print a summary of the organised dataset
print("\n" + "=" * 45)
print("Dataset organised")
print("=" * 45)
print("Processed:", processed)
print("Skipped  :", skipped)

grand_total = 0

for season in seasons:

    print(f"\n{season}")

    season_total = 0

    for image_type in image_types:

        folder = os.path.join(
            OUTPUT_DIR,
            season,
            image_type
        )

        count = sum(
            os.path.isfile(os.path.join(folder, f))
            for f in os.listdir(folder)
        )

        print(f"  {image_type}: {count}")

        season_total += count

    print(f"  Total: {season_total}")

    grand_total += season_total


print(f"\nGrand total: {grand_total}")


def read_png(filename):
    """Read an 8-bit RGB or grayscale PNG."""

    with open(filename, "rb") as f:

        # Skip the PNG signature
        f.read(8)

        width = None
        height = None
        bit_depth = None
        colour_type = None
        compressed = b""

        # Read the PNG chunks
        while True:

            length_data = f.read(4)

            if not length_data:
                break

            length = struct.unpack(">I", length_data)[0]

            chunk = f.read(4)
            data = f.read(length)

            f.read(4)  # CRC

            if chunk == b"IHDR":

                width = struct.unpack(">I", data[0:4])[0]
                height = struct.unpack(">I", data[4:8])[0]
                bit_depth = data[8]
                colour_type = data[9]

            elif chunk == b"IDAT":

                compressed += data

            elif chunk == b"IEND":

                break

    if bit_depth != 8:
        raise ValueError(
            f"Expected 8-bit PNG, got {bit_depth}-bit"
        )

    if colour_type == 2:
        channels = 3
    elif colour_type == 0:
        channels = 1
    else:
        raise ValueError(
            f"Unsupported PNG colour type: {colour_type}"
        )

    raw = zlib.decompress(compressed)

    bytes_per_pixel = channels
    row_size = width * channels

    previous = bytearray(row_size)
    pixels = []

    position = 0

    for _ in range(height):

        filter_type = raw[position]
        position += 1

        row = bytearray(
            raw[position:position + row_size]
        )

        position += row_size

        # PNG stores each row using one of five filter types.
        # Reverse the filter to recover the original pixel values.
        for x in range(row_size):

            left = (
                row[x - bytes_per_pixel]
                if x >= bytes_per_pixel
                else 0
            )

            above = previous[x]

            upper_left = (
                previous[x - bytes_per_pixel]
                if x >= bytes_per_pixel
                else 0
            )

            if filter_type == 0:

                value = row[x]

            elif filter_type == 1:

                value = row[x] + left

            elif filter_type == 2:

                value = row[x] + above

            elif filter_type == 3:

                value = row[x] + (left + above) // 2

            elif filter_type == 4:

                p = left + above - upper_left

                pa = abs(p - left)
                pb = abs(p - above)
                pc = abs(p - upper_left)

                if pa <= pb and pa <= pc:
                    predictor = left
                elif pb <= pc:
                    predictor = above
                else:
                    predictor = upper_left

                value = row[x] + predictor

            else:

                raise ValueError(
                    f"Unknown PNG filter: {filter_type}"
                )

            row[x] = value & 255

        pixels.extend(row)
        previous = row

    return width, height, channels, pixels


# Check one image from each category
print("\n" + "=" * 45)
print("Image data check")
print("=" * 45)

for season in seasons:

    print(f"\n{season}")

    for image_type in image_types:

        folder = os.path.join(
            OUTPUT_DIR,
            season,
            image_type
        )

        files = sorted(
            f
            for f in os.listdir(folder)
            if f.lower().endswith(".png")
        )

        if not files:
            print(f"\n  {image_type}: no files")
            continue

        # Use the first image as a quick check
        filename = files[0]

        filepath = os.path.join(
            folder,
            filename
        )

        try:

            width, height, channels, pixels = read_png(filepath)

            print(f"\n  {image_type}")
            print(f"    File : {filename}")
            print(f"    Size : {width} x {height}")
            print("    Type : uint8")
            print(f"    Min  : {min(pixels)}")
            print(f"    Max  : {max(pixels)}")

            if channels == 3:

                red = pixels[0::3]
                green = pixels[1::3]
                blue = pixels[2::3]

                print(
                    f"    Red  : {min(red)} to {max(red)}"
                )

                print(
                    f"    Green: {min(green)} to {max(green)}"
                )

                print(
                    f"    Blue : {min(blue)} to {max(blue)}"
                )

        except Exception as e:

            print(
                f"\n  {image_type}"
                f"\n    Could not read {filename}"
                f"\n    Error: {e}"
            )


print("\nDone.")
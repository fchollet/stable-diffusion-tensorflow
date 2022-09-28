from datetime import datetime
from tensorflow import keras
from stable_diffusion_tf.stable_diffusion import Text2Image
import argparse
from PIL import Image

parser = argparse.ArgumentParser()

parser.add_argument(
    "--prompt",
    type=str,
    nargs="?",
    default="a painting of a virus monster playing guitar",
    help="the prompt to render",
)

parser.add_argument(
    "--output",
    type=str,
    nargs="?",
    default=None,
    help="where to save the output image",
)

parser.add_argument(
    "--H",
    type=int,
    default=512,
    help="image height, in pixels",
)

parser.add_argument(
    "--W",
    type=int,
    default=512,
    help="image width, in pixels",
)

parser.add_argument(
    "--scale",
    type=float,
    default=7.5,
    help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
)

parser.add_argument(
    "--steps", type=int, default=50, help="number of ddim sampling steps"
)

parser.add_argument(
    "--seed",
    type=int,
    help="optionally specify a seed integer for reproducible results",
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=1,
    help="how many images to generate",
)

parser.add_argument(
    "--mp",
    default=False,
    action="store_true",
    help="Enable mixed precision (fp16 computation)",
)

args = parser.parse_args()


if args.mp:
    print("Using mixed precision.")
    keras.mixed_precision.set_global_policy("mixed_float16")

generator = Text2Image(img_height=args.H, img_width=args.W, jit_compile=False)
img = generator.generate(
    args.prompt,
    num_steps=args.steps,
    unconditional_guidance_scale=args.scale,
    temperature=1,
    batch_size=args.batch_size,
    seed=args.seed,
)

if not args.output:
    # When not providing an output filename, create something using
    # the prompt and a timestamp to prevent overwriting of existing files
    # Get a timestamp without  microseconds, and replace colons with dots
    timestamp = datetime.now().isoformat("T").split(".")[0].replace(":", ".")

    # Create a 'slug' with only valid alphanumeric characters and spaces to
    # prevent filename issues
    slug_prompt = "".join(c for c in args.prompt if (c.isalnum() or c in "_- "))

    # And create the final filename
    fname = f"{timestamp} - {slug_prompt}"
else:
    # Output filename provided, use that
    fname = args.output

if fname.endswith(".png"):
    fname = fname[:-4]

if args.batch_size == 1:
    Image.fromarray(img[0]).save(f"{fname}.png")
    print(f"saved at {fname}.png")
else:
    for i in range(args.batch_size):
        fname_i = f"{fname}_{i}.png"
        Image.fromarray(img[i]).save(fname_i)
        print(f"saved at {fname_i}")

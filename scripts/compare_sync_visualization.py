import argparse
import numpy as np
import hdf5plugin  # this is needed for h5py to work correctly
import matplotlib.pyplot as plt
from pathlib import Path
from dsec_det.dataset import DSECDet, render_object_detections_on_image, render_events_on_image


def to_gamma_corrected(image):
    image = (255 * (image.astype("float32") / 255) ** (1 / 2.2)).astype("uint8")
    return image


def visualize_sync_difference(dataset_root, split, start_idx, end_idx, output_dir=None):
    front_ds = DSECDet(dataset_root, split=split, sync="front")
    back_ds = DSECDet(dataset_root, split=split, sync="back")

    for idx in range(start_idx, end_idx):
        try:
            front = {
                'image': to_gamma_corrected(front_ds.get_image(idx).copy()),
                'events': front_ds.get_events(idx),
                'tracks': front_ds.get_tracks(idx),
            }
            back = {
                'image': to_gamma_corrected(back_ds.get_image(idx).copy()),
                'events': back_ds.get_events(idx),
                'tracks': back_ds.get_tracks(idx),
            }

            if not np.allclose(front['image'], back['image']):
                print(f"Image mismatch at index {idx}")

            combined_img = front['image'].copy()
            if front['tracks'].size > 0:
                front_tracks = front['tracks'].copy()
                front_tracks['class_id'][:] = 0
                combined_img = render_object_detections_on_image(
                    combined_img, front_tracks, label="front"
                )

            if back['tracks'].size > 0:
                back_tracks = back['tracks'].copy()
                back_tracks['class_id'][:] = 1
                combined_img = render_object_detections_on_image(
                    combined_img, back_tracks, label="back"
                )

            front_events_img = front['image'].copy()
            back_events_img = back['image'].copy()

            front_events_img = render_events_on_image(front_events_img, front['events']['x'], front['events']['y'],
                                                      front['events']['p'])
            back_events_img = render_events_on_image(back_events_img, back['events']['x'], back['events']['y'],
                                                     back['events']['p'])

            fig, axs = plt.subplots(1, 3, figsize=(3 * (combined_img.shape[1] / 100), combined_img.shape[0] / 100),
                                    dpi=100)

            axs[0].imshow(combined_img[..., ::-1])
            axs[0].set_title("back=1(cyan), front=0(blue)", fontsize=8)

            axs[1].imshow(front_events_img[..., ::-1])
            axs[1].set_title("Front Events", fontsize=8)

            axs[2].imshow(back_events_img[..., ::-1])
            axs[2].set_title("Back Events", fontsize=8)

            for ax in axs:
                ax.axis('off')

            fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.01, hspace=0.01)

            if output_dir:
                out_path = Path(output_dir) / f"{idx:04d}.png"
                fig.savefig(str(out_path), bbox_inches='tight', pad_inches=0)
                print(f"[✓] Saved {out_path}")
                plt.close(fig)
            else:
                plt.show()

        except Exception as e:
            print(f"[!] Failed at index {idx}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Visualize front vs back sync differences in DSEC")
    parser.add_argument("--dsec_merged", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--start", type=int, default=10)
    parser.add_argument("--end", type=int, default=90)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    visualize_sync_difference(args.dsec_merged, args.split, args.start, args.end, args.output_dir)

import unittest

import numpy as np
import pandas as pd

from warehouse_sim.visualization import (
    plot_warehouse_animation,
    plot_warehouse_animation_3d,
)


def sample_trace() -> pd.DataFrame:
    records = []
    for time, worker_x, worker_state, wait_total, completed in (
        (0.0, 0.0, "moving", 0.0, 0),
        (1.0, 1.0, "moving", 2.0, 1),
        (2.0, 2.0, "picking", 2.0, 1),
    ):
        records.extend(
            [
                {
                    "time": time,
                    "agent_id": "worker-1",
                    "agent_type": "worker",
                    "x": worker_x,
                    "y": 0.0,
                    "state": worker_state,
                    "wait_time_total": wait_total,
                    "queue_count": 1 if time == 1.0 else 0,
                    "completed_order_count": completed,
                },
                {
                    "time": time,
                    "agent_id": "forklift-1",
                    "agent_type": "forklift",
                    "x": 2.0,
                    "y": 1.0,
                    "state": "waiting",
                    "wait_time_total": wait_total,
                    "queue_count": 1 if time == 1.0 else 0,
                    "completed_order_count": completed,
                },
            ]
        )
    return pd.DataFrame(records)


class WarehouseAnimationTests(unittest.TestCase):
    def test_creates_browser_animation_controls_and_synchronized_kpis(self) -> None:
        figure = plot_warehouse_animation(
            np.array([["I", ".", "S"], ["A", "A", "."]]),
            sample_trace(),
            fps=20,
            max_frames=3,
        )

        self.assertEqual(len(figure.frames), 3)
        self.assertEqual(len(figure.layout.sliders[0].steps), 3)
        button_labels = [
            button.label
            for menu in figure.layout.updatemenus
            for button in menu.buttons
        ]
        self.assertEqual(button_labels, ["▶ 再生", "⏸ 一時停止"])
        self.assertEqual(list(figure.data[-2].x), [0.0])
        self.assertEqual(list(figure.data[-1].x), [2.0])

        middle_frame = figure.frames[1]
        self.assertEqual(list(middle_frame.data[0].x), [1.0])
        annotations = [annotation.text for annotation in middle_frame.layout.annotations]
        self.assertTrue(any("稼働中の作業員" in text and "1" in text for text in annotations))
        self.assertTrue(any("累積待ち時間" in text and "2.0" in text for text in annotations))
        self.assertTrue(
            any(
                "待ち要求数（渋滞指標） / 完了オーダー" in text
                and "1 / 1" in text
                for text in annotations
            )
        )

    def test_caps_frame_count(self) -> None:
        figure = plot_warehouse_animation(
            np.array([["I", "S"]]),
            sample_trace(),
            max_frames=2,
        )
        self.assertEqual(len(figure.frames), 2)

    def test_creates_3d_animation_and_scales_playback_speed(self) -> None:
        figure = plot_warehouse_animation_3d(
            np.array([["I", ".", "S"], ["A", "A", "."]]),
            sample_trace(),
            fps=20,
            max_frames=3,
            speed_multiplier=2.0,
        )

        self.assertEqual(len(figure.frames), 3)
        self.assertEqual(len(figure.layout.sliders[0].steps), 3)
        self.assertEqual(figure.layout.updatemenus[0].buttons[0].label, "▶ 再生")
        self.assertEqual(figure.layout.updatemenus[0].buttons[1].label, "⏸ 一時停止")
        self.assertEqual(figure.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"], 25)
        self.assertEqual(figure.layout.scene.camera.eye.x, 0.0)
        self.assertEqual(figure.layout.scene.camera.eye.y, 0.0)
        self.assertEqual(figure.layout.scene.camera.eye.z, 2.5)
        self.assertEqual(figure.data[-2].marker.size, 13)
        self.assertEqual(figure.data[-2].marker.color, "#168bff")
        self.assertEqual(figure.data[-1].marker.size, 13)
        self.assertEqual(figure.data[-1].marker.color, "#ff9c2a")
        middle_frame = figure.frames[1]
        self.assertEqual(list(middle_frame.data[0].x), [1.0])
        self.assertEqual(list(middle_frame.data[0].y), [1.0])
        self.assertEqual(list(middle_frame.data[0].z), [0.5])
        self.assertEqual(len(middle_frame.layout.annotations), 4)

    def test_3d_animation_rejects_unsupported_speed(self) -> None:
        with self.assertRaisesRegex(ValueError, "speed_multiplier"):
            plot_warehouse_animation_3d(
                np.array([["I"]]),
                sample_trace(),
                speed_multiplier=3.0,
            )

    def test_rejects_invalid_trace_and_settings(self) -> None:
        with self.assertRaisesRegex(ValueError, "fps"):
            plot_warehouse_animation(np.array([["I"]]), sample_trace(), fps=0)
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            plot_warehouse_animation(np.array([["I"]]), pd.DataFrame())


if __name__ == "__main__":
    unittest.main()

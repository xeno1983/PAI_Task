"""Live difficulty of the training task: raise it as the score allows, ease it when scores stay low."""

from __future__ import annotations

from pick_and_carry.curriculum import CATCH_LEVELS, LOCOMOTION_STAGES, retreat_choice


class Difficulty:
    """Raise the live task as the score allows, and ease it when scores stay low."""

    def __init__(self, curriculum: str, catch_level: str):
        self.poor = 0
        self.good = 0
        self.kind = None
        self.current = 0
        self.target = 0
        self.labels: tuple[str, ...] = ()
        if curriculum in LOCOMOTION_STAGES:
            # Start on flat ground and climb to the selected bumps once walking is tracked.
            self.kind = "terrain"
            self.labels = ("平地", "半分の段差", "選んだ不整地")
            self.target = 2
            self.current = 0
        elif curriculum == "catch":
            self.kind = "catch"
            self.labels = ("静止", "ゆっくり", "ばらす", "飛ぶ", "ノイズ")
            self.target = CATCH_LEVELS.index(catch_level)
            self.current = self.target

    def start(self, env) -> None:
        """Put the environment on the starting rung. Walk begins on flat ground."""
        if self.kind == "terrain":
            env.set_terrain_mix(self.current / max(1, self.target))

    def update(self, env, score: float | None) -> str:
        if self.kind is None:
            return ""
        new, self.poor, self.good = retreat_choice(
            score,
            current=self.current,
            target=self.target,
            poor_streak=self.poor,
            good_streak=self.good,
        )
        if new != self.current:
            previous = self.current
            self.current = new
            if self.kind == "terrain":
                env.set_terrain_mix((0.0, 0.5, 1.0)[new])
            else:
                env.set_catch_level(CATCH_LEVELS[new])
            if new < previous:
                print(f"成績が低いので、{self.labels[new]} に戻しました。回復したら上げます。", flush=True)
            else:
                print(f"成績が戻ったので、{self.labels[new]} に上げました。", flush=True)
        if self.current == self.target:
            return ""
        return self.labels[self.current]

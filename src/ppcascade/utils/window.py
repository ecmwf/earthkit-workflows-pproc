import bisect
from dataclasses import dataclass, field


@dataclass
class Range:
    name: str
    steps: list[int]


@dataclass
class WindowConfig:
    operation: str
    include_init: bool
    options: dict
    ranges: list[Range] = field(default_factory=list)

    # TODO: precomputed windows, how is their EFI computed?? Do we need num steps?
    def add_range(
        self,
        start: int,
        end: int,
        step: int | None = None,
        allowed_steps: list | None = None,
    ):
        window_size = end - start
        if window_size == 0:
            name = str(end)
            steps = [start]
        else:
            name = f"{start}-{end}"

            # Set steps in window
            if step != None:
                range_start = start if self.include_init else start + step
                steps = list(range(range_start, end + 1, step))
            else:
                assert allowed_steps != None or len(allowed_steps) != 0

                if self.include_init:
                    start_index = allowed_steps.index(start)
                else:
                    # Case when window.start not in steps
                    start_index = bisect.bisect_right(allowed_steps, start)
                steps = allowed_steps[start_index : allowed_steps.index(end) + 1]

        assert name not in self.ranges
        self.ranges.append(Range(name, steps))

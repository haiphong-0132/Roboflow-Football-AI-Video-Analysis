from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class SoccerPitchConfiguration:
    width: int = 7000
    length: int = 12000
    penalty_box_width: int = 4100
    penalty_box_length: int = 2015
    goal_box_width: int = 1832
    goal_box_length: int = 550
    centre_circle_radius: int = 915
    penalty_spot_distance: int = 1100

    @property
    def vertices(self) -> List[Tuple[int, int]]:
        return [
            (0, 0),
            (0, (self.width - self.penalty_box_width) / 2),
            (0, (self.width - self.goal_box_width) / 2),
            (0, (self.width + self.goal_box_width) / 2),
            (0, (self.width + self.penalty_box_width) / 2),
            (0, self.width),

            (self.goal_box_length, (self.width - self.goal_box_width) / 2),
            (self.goal_box_length, (self.width + self.goal_box_width) / 2),
            
            (self.penalty_spot_distance, self.width / 2),

            (self.penalty_box_length, (self.width - self.penalty_box_width) / 2),
            (self.penalty_box_length, (self.width - self.goal_box_width) / 2),
            (self.penalty_box_length, (self.width + self.goal_box_width) / 2),
            (self.penalty_box_length, (self.width + self.penalty_box_width) / 2),

            (self.length / 2, 0),
            (self.length / 2, self.width / 2 - self.centre_circle_radius),
            (self.length / 2, self.width / 2 + self.centre_circle_radius),
            (self.length / 2, self.width),

            (self.length - self.penalty_box_length, (self.width - self.penalty_box_width) / 2),
            (self.length - self.penalty_box_length, (self.width - self.goal_box_width) / 2),
            (self.length - self.penalty_box_length, (self.width + self.goal_box_width) / 2),
            (self.length - self.penalty_box_length, (self.width + self.penalty_box_width) / 2),

            (self.length - self.penalty_spot_distance, self.width / 2),

            (self.length - self.goal_box_length, (self.width - self.goal_box_width) / 2),
            (self.length - self.goal_box_length, (self.width + self.goal_box_width) / 2),

            (self.length, 0),
            (self.length, (self.width - self.penalty_box_width) / 2),
            (self.length, (self.width - self.goal_box_width) / 2),
            (self.length, (self.width + self.goal_box_width) / 2),
            (self.length, (self.width + self.penalty_box_width) / 2),
            (self.length, self.width),

            (self.length / 2 - self.centre_circle_radius, self.width / 2),
            (self.length / 2 + self.centre_circle_radius, self.width / 2),
        ]
    
    edges: List[Tuple[int, int]] = field(default_factory=lambda: [
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (7, 8), 
        (10, 11), (11, 12), (12, 13), (14, 15), (15, 16),
        (16, 17), (18, 19), (19, 20), (20, 21), (23, 24),
        (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
        
        (1, 14), (2, 10), (3, 7), (4, 8), (5, 13), (6, 17),
        (14, 25), (18, 26), (23, 27), (24, 28), (21, 29), (17, 30)
    ])

    labels: List[str] = field(default_factory=lambda: [
        "01", "02", "03", "04", "05", "06", "07", "08",
        "09", "10", "11", "12", "13", "15", "16", "17", 
        "18", "20", "21", "22", "23", "24", "25", "26", 
        "27", "28", "29", "30", "31", "32", "14", "19"
    ])

    colors: List[str] = field(default_factory=lambda: [
        '#FF1493', '#FF1493', '#FF1493', '#FF1493', '#FF1493', '#FF1493', '#FF1493', '#FF1493',
        '#FF1493', '#FF1493', '#FF1493', '#FF1493', '#FF1493', "#00BFFF", "#00BFFF", "#00BFFF",
        "#00BFFF", "#00BFFF", '#FF6347', '#FF6347', '#FF6347', '#FF6347', '#FF6347', '#FF6347', 
        '#FF6347', '#FF6347', '#FF6347', '#FF6347', '#FF6347', '#FF6347', '#FF6347', '#00BFFF',
        '#00BFFF'
    ])
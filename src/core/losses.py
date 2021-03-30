import torch
import torch.nn as nn

class DistortionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, distortion, coordinate_norms):
        dst_coordinates = []
        for x, y in coordinate_norms:
            dst_x, dst_y = self.project(x, y, distortion)
            dst_coordinates.append((dst_x, dst_y))

        return torch.mean(torch.pow(self.calc_distance(dst_coordinates), 2))

    def project(self, x, y, distortion):
        a = distortion * x * (1 + y**2/x**2)
        c = -x

        new_x = (-1 + torch.sqrt(1 - 4*a*c)) / (2*a)
        new_y = (y/x) * new_x

        return new_x, new_y

    def calc_distance(self, coordinates):
        return 100 * (coordinates[1][0] - coordinates[0][0])

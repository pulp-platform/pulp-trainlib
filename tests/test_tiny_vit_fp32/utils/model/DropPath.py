from timm.models.layers import DropPath as TimmDropPath


class DropPath(TimmDropPath):
    def __init__(self, drop_prob=None):
        super().__init__(drop_prob=drop_prob)

        self.drop_prob = drop_prob

    def __repr__(self):
        msg = super().__repr__()
        msg += f"(drop_prob={self.drop_prob})"

        return msg

"""Figures for the paper companion: the homunculus and the workflow ladder."""

from .homunculus import distortion_profile, homunculus_figure, homunculus_pair
from .ladder import RUNGS, ladder_svg

__all__ = [
    "RUNGS",
    "distortion_profile",
    "homunculus_figure",
    "homunculus_pair",
    "ladder_svg",
]

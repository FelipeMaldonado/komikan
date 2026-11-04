"""Komikan: the Mapuche board game, as a rules engine, Gymnasium environment and web game."""

from gymnasium.envs.registration import register

register(id="Komikan-v0", entry_point="komikan.env:KomikanEnv")

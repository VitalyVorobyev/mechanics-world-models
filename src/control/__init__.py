"""Closed-loop control: task reward, MPPI planner and dynamics backends.

The planner is backend-agnostic: a backend scores a batch of action
sequences from the current state (true MuJoCo model, or a learned model).
"""

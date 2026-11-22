"""One shared head arrival tolerance, expressed in physical encoder ticks."""
# Operator accepts the observed 0.88-degree load offset. Use a one-degree
# arrival band, rounded DOWN so the integer threshold never exceeds 1 degree.
# 4096 ticks/revolution: 11 ticks = 0.967 degrees; 12 ticks = 1.055 degrees.
TARGET_TOLERANCE_TICKS = int(1.0 * 4096 / 360)
TARGET_TOLERANCE_DEG = TARGET_TOLERANCE_TICKS * 360 / 4096

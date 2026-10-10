"""Shared report templates and styles for PDF generation."""

# Color palette
COLOR_PRIMARY = (37, 99, 235)  # blue-600
COLOR_SECONDARY = (30, 64, 175)  # blue-800
COLOR_ACCENT = (59, 130, 246)  # blue-500
COLOR_BG_LIGHT = (239, 246, 255)  # blue-50
COLOR_TEXT = (31, 41, 55)  # gray-800
COLOR_TEXT_MUTED = (107, 114, 128)  # gray-500
COLOR_BORDER = (209, 213, 219)  # gray-300
COLOR_WHITE = (255, 255, 255)
COLOR_GREEN = (16, 185, 129)
COLOR_YELLOW = (245, 158, 11)
COLOR_RED = (239, 68, 68)

# Severity colors for badges
SEVERITY_COLORS = {
    "Little or No Damage": (16, 185, 129),
    "Mild Damage": (245, 158, 11),
    "Severe Damage": (239, 68, 68),
}

# Priority colors
PRIORITY_COLORS = {
    "Low": (16, 185, 129),
    "Medium": (245, 158, 11),
    "High": (239, 68, 68),
}

# Section titles
SECTION_TITLES = {
    "incident": "Incident Information",
    "prediction": "Prediction Summary",
    "assessment": "Emergency Assessment",
    "actions": "Recommended Actions",
    "technical": "Technical Details",
}

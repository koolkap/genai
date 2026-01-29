from PIL import Image, ImageDraw
import pandas as pd
import os

# -------------------------
# Create folders
# -------------------------
os.makedirs("data/text", exist_ok=True)
os.makedirs("data/csv", exist_ok=True)
os.makedirs("data/images", exist_ok=True)

# -------------------------
# 1. Dummy TEXT file
# -------------------------
text_content = """
Company Policy Document

1. Employees must work 8 hours per day.
2. Remote work is allowed twice a week.
3. All data must be handled securely.
4. AI tools should respect user privacy.
"""

with open("data/text/company_policy.txt", "w") as f:
    f.write(text_content)

# -------------------------
# 2. Dummy CSV file
# -------------------------
df = pd.DataFrame({
    "employee_id": [1, 2, 3],
    "name": ["Alice", "Bob", "Charlie"],
    "role": ["Engineer", "Designer", "Manager"],
    "salary": [70000, 65000, 90000]
})

df.to_csv("data/csv/employees.csv", index=False)

# -------------------------
# 3. Dummy IMAGE file
# -------------------------
img = Image.new("RGB", (400, 300), color="white")
draw = ImageDraw.Draw(img)
draw.text((50, 130), "Office Meeting Scene", fill="black")

img.save("data/images/office_scene.png")

print("Dummy data generated successfully!")

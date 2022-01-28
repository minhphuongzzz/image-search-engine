import numpy as np
from PIL import Image
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path
from search import search_image
app = Flask(__name__)

# Read image features

# features = []
# img_paths = []
# for feature_path in Path("./data/feature_database/color").glob("*.npy"):
#     features.append(np.load(feature_path))
#     img_paths.append(Path("./data/img") / (feature_path.stem + ".jpg"))
# features = np.array(features)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']

        # Save query image
        img = Image.open(file.stream)  # PIL image
        img = np.array(img)

        # Run search
        output_paths, _ = search_image(img, 'hog', 'cosine', 20)
        print("haha", output_paths)
        return render_template('index.html',
                               paths = output_paths)
    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run("0.0.0.0")
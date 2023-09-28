from flask import Flask, request, jsonify
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN
app = Flask(__name__)


def draw_image_with_boxes(image, result_list):
    count = 0
    data = pyplot.imread(image)
    pyplot.imshow(data)
    ax = pyplot.gca()
    for result in result_list:
        x, y, width, height = result['box']
        rect = Rectangle((x, y), width, height, fill=False, color='red')
        count = count + 1
        ax.add_patch(rect)
        for key, value in result['keypoints'].items():
            dot = Circle(value, radius=2, color='red')
            ax.add_patch(dot)
    return count


def fun(image):
    pixels = pyplot.imread(image)
    detector = MTCNN()
    faces = detector.detect_faces(pixels)
    answer = (draw_image_with_boxes(image, faces))
    return answer


@app.route('/upload image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    file.save(file.filename)
    a = str(fun(file))
    s = "No Of Faces Detected :- "+a
    return jsonify(s)


if __name__ == '__main__':
    app.run(debug=True)

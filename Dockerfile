FROM python:3.8
# Set the working directory in the container
WORKDIR /Heart Disease Prediction

# Copy the current directory contents into the container at/app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run app.py when the container launches
CMD ["python", "./Heart Disease Prediction.pynb"]
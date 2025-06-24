import numpy as np

def sample_rot_mats(hand_dir_x_w, num_samples, visible_points_w):
    # Sample num_samples perpendicular unit vectors
    reference_vector = hand_dir_x_w.reshape(3)
    if np.abs(reference_vector[0]) < 0.9:
        temp_vector = np.array([1, 0, 0])
    else:
        temp_vector = np.array([0, 1, 0])
    # Create a vector perpendicular to the reference vector
    first_perpendicular = np.cross(reference_vector, temp_vector)
    first_perpendicular = first_perpendicular / np.linalg.norm(first_perpendicular)
    # Create another perpendicular vector to form an orthogonal basis
    second_perpendicular = np.cross(reference_vector, first_perpendicular)
    second_perpendicular = second_perpendicular / np.linalg.norm(second_perpendicular)
    
    # Sample num_samples vectors evenly distributed around the circle
    thetas = np.linspace(0, 2*np.pi, num_samples, endpoint=False)
    perpendicular_vectors = np.zeros((num_samples, 3))
    for j, theta in enumerate(thetas):  
        perpendicular_vectors[j] = first_perpendicular * np.cos(theta) + second_perpendicular * np.sin(theta)
        perpendicular_vectors[j] = perpendicular_vectors[j] / np.linalg.norm(perpendicular_vectors[j])
        if perpendicular_vectors[j,1] < 0:
            perpendicular_vectors[j] = -perpendicular_vectors[j]
    # Calculate projection lengths of visible points along each perpendicular direction
    centered_points = visible_points_w - np.mean(visible_points_w.reshape(200,3), axis=0)
    projection_lengths = np.zeros((num_samples, 200))
    for j in range(num_samples):
        projection_lengths[j] = np.dot(centered_points, perpendicular_vectors[j])
    
    # Calculate statistics for each projection direction
    projection_min = np.min(projection_lengths, axis=1)
    projection_max = np.max(projection_lengths, axis=1)
    projection_range = projection_max - projection_min

    y_dir_per_sample = np.zeros((num_samples, 3))
    rot_mat_per_sample = np.zeros((num_samples, 3, 3))
    for j in range(num_samples):
        y_dir_per_sample[j] = np.cross(hand_dir_x_w, perpendicular_vectors[j])
        y_dir_per_sample[j] = y_dir_per_sample[j] / np.linalg.norm(y_dir_per_sample[j])
        rot_mat_per_sample[j] = -np.stack((hand_dir_x_w.reshape(3), y_dir_per_sample[j].reshape(3), perpendicular_vectors[j].reshape(3)), axis=-1)

    return rot_mat_per_sample, projection_range    
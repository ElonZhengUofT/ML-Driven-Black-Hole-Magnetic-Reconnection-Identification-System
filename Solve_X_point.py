import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# Define the magnetic field function (2D example)
def magnetic_field(pos):
    """
    Returns the Bx and By components of the magnetic field at a given position.
    This example uses a simple dipole model:
    Bx = x - x^3
    By = -y + y^3
    """
    x, y = pos
    Bx = x - x**3
    By = -y + y**3
    return [Bx, By]

# Calculate the Jacobian matrix
def jacobian(pos):
    """
    Calculates the Jacobian matrix of the magnetic field at a given position.
    """
    x, y = pos
    dBx_dx = 1 - 3*x**2
    dBx_dy = 0
    dBy_dx = 0
    dBy_dy = -1 + 3*y**2
    return np.array([[dBx_dx, dBx_dy],
                     [dBy_dx, dBy_dy]])

# Find null points of the magnetic field
def find_null_points(initial_guesses):
    """
    Uses fsolve to find null points of the magnetic field.
    initial_guesses: list of initial guesses for positions
    Returns a list of found null points
    """
    null_points = []
    for guess in initial_guesses:
        sol, infodict, ier, mesg = fsolve(magnetic_field, guess, full_output=True)
        if ier == 1:  # Solution found successfully
            # Check for duplicate points
            if not any(np.allclose(sol, np.array(p), atol=1e-4) for p in null_points):
                null_points.append(sol)
    return null_points

# Classify null points as X-point or O-point
def classify_null_point(pos):
    """
    Classifies a null point based on the eigenvalues of the Jacobian matrix.
    Returns 'X-point' or 'O-point'
    """
    J = jacobian(pos)
    eigenvalues = np.linalg.eigvals(J)
    if np.all(np.isreal(eigenvalues)):
        if np.prod(eigenvalues) < 0:
            return 'X-point'
    return 'O-point'

# Main program
def main():
    # Define initial guesses (can be adjusted based on the specific problem)
    initial_guesses = [
        [0.0, 0.0],
        [1.0, 1.0],
        [-1.0, 1.0],
        [1.0, -1.0],
        [-1.0, -1.0]
    ]

    # Find null points
    null_points = find_null_points(initial_guesses)
    print("Found null points:")
    for p in null_points:
        point_type = classify_null_point(p)
        print(f"Position: ({p[0]:.4f}, {p[1]:.4f}) Type: {point_type}")

    # Visualization of magnetic field and null points
    x = np.linspace(-2, 2, 400)
    y = np.linspace(-2, 2, 400)
    X, Y = np.meshgrid(x, y)
    Bx = X - X**3
    By = -Y + Y**3

    plt.figure(figsize=(8,8))
    plt.streamplot(X, Y, Bx, By, density=1.5, linewidth=1, arrowsize=1, arrowstyle='->')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Magnetic Field Lines')

    # Mark null points
    for p in null_points:
        point_type = classify_null_point(p)
        if point_type == 'X-point':
            plt.plot(p[0], p[1], 'ro', label='X-point')
        else:
            plt.plot(p[0], p[1], 'bo', label='O-point')

    # Avoid duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.grid()
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    main()

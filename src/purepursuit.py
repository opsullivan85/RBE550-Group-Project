import numpy as np
import itertools

# it would be nice to cache this, but np.ndarrays
# don't like being hashed
def subdivide(path: np.ndarray, num: int) -> np.ndarray:
    """Subdivide each section of a path into a certian number of points

    Args:
        path (Path): Path to subdivide.
            shape (n, m), where m is the dimensionality of the system.
        num (int): Number of points per subdivided line.
            minimum of 2 (ie no subdivision)

    Returns:
        Path: A subdivided path
        
    Examples:
        >>> import numpy as np
        >>> a = np.asarray((0,0))
        >>> b = np.asarray((2,2))
        >>> ab = np.vstack((a,b))
        >>> subdivide(ab, 1)
        Traceback (most recent call last):
        ...
        AssertionError: num must be atleast 2
        >>> subdivide(ab, 2)
        array([[0., 0.],
               [2., 2.]])
        >>> subdivide(ab, 3)
        array([[0., 0.],
               [1., 1.],
               [2., 2.]])
    """
    assert num >= 2, "num must be atleast 2"
    sections: list[np.ndarray] = [path[0]]
    for start, end in itertools.pairwise(path):
        # subdivide, exclude the first of every subdivision
        # to avoid double counting points
        sections.append(np.linspace(start=start, stop=end, num=num)[1:])
        
    return np.vstack(sections)

def pure_pursuit(path: np.ndarray, agent_position: np.ndarray, radius: float, subdivision_num: int = 10) -> np.ndarray:
    """Pure pursuit controller

    Args:
        path (np.ndarray): Path to follow.
            shape (n, m), where m is the dimensionality of the system.
        agent_position (np.ndarray): Position of the agent.
            shape (m,)
        radius (float): Lookahead distance for the algorithm.
            defined as a radius in `m` dimensional space.
        subdivision_num (int): Number of points per subdivided line.
            minimum of 2 (ie no subdivision). Default is 10.

    Returns:
        np.ndarray: A vector from the `agent_position` to the pure_pursuit target.
            Shape is (m,). Not normalized, but will usually have magnatude ~= `radius`,
            with some exceptions
    """
    subdivision = subdivide(path=path, num=subdivision_num)
    offsets = subdivision - agent_position
    distances = np.linalg.norm(offsets, axis=1) - radius
    
    # if the end is close enough, just path there
    if distances[-1] < 0:
        return offsets[-1]
    
    # indicies of sign changes. Each sign change corresponds to 
    # an intersection of the circle with r=`radius` with the path
    # snippet from https://stackoverflow.com/a/44198074
    sign_changes = np.where(np.sign(distances[:-1]) != np.sign(distances[1:]))[0] + 1
    
    if len(sign_changes) == 0:
        # we are more than a radius away from the path.
        # just go back towards the closest point
        return offsets[np.argmin(distances)]
    else:
        # return the furthest along 
        return offsets[sign_changes[-1]]
    


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    
    import matplotlib.pyplot as plt
    
    a = np.asarray((0,0))
    b = np.asarray((2,2))
    c = np.asarray((2,5))
    d = np.asarray((5,1))
    e = np.asarray((3,-5))
    path = np.vstack((a,b,c,d,e))
    
    agent_position = np.asarray((0, 0.5))
    
    followed_path = []
    while(np.linalg.norm(agent_position-path[-1]) > 0.2):
        followed_path.append(agent_position.copy())
        control_vector = pure_pursuit(path=path, agent_position=agent_position, radius=1, subdivision_num=100)
        unit_control = control_vector / np.linalg.norm(control_vector)
        agent_position += 0.1 * unit_control
    
    followed_path = np.vstack(followed_path)
    plt.scatter(followed_path[:, 0], followed_path[:, 1])
    plt.plot(*zip(*path), '-', color="grey")
    plt.plot(*zip(*path), 'o', color="black")
    plt.savefig("purepursuit_test")
# TSP-Algo

```markdown
# Traveling Salesperson Problem Solver

This Streamlit application visualizes and computes optimized routes using the Traveling Salesperson Problem (TSP) approach, leveraging various algorithms to find the most efficient path through multiple cities.

## Features

- Interactive map visualization with Folium
- Integration of geodesic calculations to determine distances
- Multiple TSP algorithms including Nearest Neighbor, Random Sampling, Genetic Algorithm, and Simulated Annealing
- Custom CSS for enhanced UI
- Option to load sample data or input new data dynamically

   ```
## Installation

To run this application, you will need Python installed on your system, along with several libraries. Here's how you can set it up:

1. Clone the repository:
   
   git clone Link


3. Install the required packages:
   ```bash
   pip install streamlit folium numpy pandas geopy
   ```

## Usage

To run the application, navigate to the project directory and run the following command:

```bash
streamlit run TSP.py
```

This will start the Streamlit application and open it in your default web browser.

## Algorithms

This application includes a range of algorithms to solve the Traveling Salesperson Problem (TSP), each with unique approaches and benefits:

- **Nearest Neighbor**: A simple heuristic that selects the nearest unvisited city, providing a quick but potentially suboptimal path.
- **Random Sampling**: Evaluates random permutations of the cities and selects the path with the shortest distance, offering a straightforward stochastic approach.
- **Genetic Algorithm**: Applies principles of natural selection, including crossover and mutation, to evolve solutions towards better routes.
- **Simulated Annealing**: Uses the concept of annealing in metallurgy to escape local minima and find better solutions over time by gradually reducing the "temperature" of the solution space.
- **Christofides' Algorithm**: An approximation algorithm that guarantees to come within a factor of 1.5 of the optimal length, best used for symmetric TSP where the distances comply with the triangle inequality.
- **Ant Colony Optimization**: A probabilistic technique inspired by the behavior of ants seeking paths between their colony and food source, using pheromone trails to find shorter paths over time.

These algorithms offer a blend of exact, approximate, and heuristic methods, allowing users to compare performance and effectiveness across different approaches.


## Customization

You can modify the `city_names` dictionary in the `TSP.py` file to include other cities or change the existing ones. Additionally, the styling can be adjusted via the CSS in the `add_custom_css` function.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your features or fixes.

## License

Distributed under the MIT License. See `LICENSE` for more information.



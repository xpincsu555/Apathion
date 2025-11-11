# Apathion Gameplay Guide

## Quick Start

Run an interactive game session:

```bash
apathion play --algorithm=astar --map_type=simple --waves=5 --enemies=10
```

Or use a configuration file:

```bash
apathion play --config_file=configs/example_game.json
```

## Controls

### Keyboard Controls

- **ESC**: Quit the game
- **Space**: Pause/Resume the game
- **Tab**: Toggle visualization mode (Minimal → Normal → Debug)
- **T**: Cycle through available tower types
- **1-4**: Directly select tower type (1=Basic, 2=Sniper, 3=Rapid, 4=Area)

### Mouse Controls

- **Left Click**: Place a tower at the clicked grid position
  - Tower will only be placed if the position is walkable
  - Enemy paths are automatically recalculated when towers are placed

## Visualization Modes

### Minimal Mode
Displays only the essential game elements:
- Map grid
- Towers
- Enemies

### Normal Mode (Default)
Includes everything from Minimal mode plus:
- Health bars above enemies
- Tower attack ranges (blue circles)
- Game statistics overlay (top-left)
- Current wave, time, and enemy counts

### Debug Mode
Includes everything from Normal mode plus:
- Enemy planned paths (yellow lines)
- Pheromone trails (for ACO algorithm - yellow/red gradient)
- FPS counter
- Frame timing information

## Game Elements

### Towers

**Basic Tower**
- Damage: 20
- Range: 3 cells
- Attack Rate: 1.0 per second
- Balanced all-around tower

**Sniper Tower**
- Damage: 50
- Range: 6 cells
- Attack Rate: 0.5 per second
- Long-range, high damage

**Rapid Tower**
- Damage: 10
- Range: 2.5 cells
- Attack Rate: 3.0 per second
- Fast firing, short range

**Area Tower**
- Damage: 15
- Range: 4 cells
- Attack Rate: 0.75 per second
- Medium range area effect

### Enemies

**Normal Enemy**
- Health: 100
- Speed: 1.0 cells/second
- Standard enemy unit

**Fast Enemy**
- Health: 60
- Speed: 2.0 cells/second
- Quick but fragile

**Tank Enemy**
- Health: 200
- Speed: 0.5 cells/second
- Slow but durable

## Pathfinding Algorithms

### A* (A-Star)
- Finds optimal paths considering distance and tower damage
- Quick and efficient
- Good baseline performance

### ACO (Ant Colony Optimization)
- Uses pheromone trails to discover paths
- Adapts over time as enemies find successful routes
- **Visualization**: Yellow to red pheromone overlay in Debug mode
- Multiple enemies collaborate to find safer paths

### DQN (Deep Q-Network)
- Machine learning-based pathfinding
- Learns from experience
- Requires pre-trained model (use `apathion train` command)

## Map Types

### Simple Map
- Open 30x20 grid
- Direct path from spawn to goal
- Good for testing and learning

### Branching Map
- Multiple path options
- Forces strategic decisions
- Tests algorithm adaptability

### Open Arena
- 40x30 large open space
- Maximum routing freedom
- Best for complex tower arrangements

## Game Flow

1. **Wave Spawning**: Enemies spawn at the designated spawn point with a delay between each enemy (default 1 second)
2. **Pathfinding**: Each enemy calculates its path using the selected algorithm
3. **Movement**: Enemies move along their planned paths
4. **Combat**: Towers attack enemies within range
5. **Wave Completion**: When all enemies are defeated or escaped, the next wave spawns
6. **Victory**: Complete all waves with minimal enemies escaping
7. **Defeat**: Too many enemies reach the goal (20+ escapes)

## Tips & Strategies

### Tower Placement
- Place towers early in the game before too many waves spawn
- Consider enemy paths when placing towers
- Create chokepoints to maximize tower coverage
- Mix tower types for varied attack patterns

### Using ACO Effectively
- Watch pheromone trails in Debug mode
- High pheromone areas (red) indicate popular enemy routes
- Place new towers on high-pheromone paths to disrupt enemy flow
- Pheromones evaporate over time, so patterns shift

### Performance
- Lower FPS if experiencing lag (edit config file `target_fps`)
- Reduce enemy count for smoother gameplay
- Disable visual features in config for better performance

## Configuration

Customize your game by editing `configs/example_game.json` or creating your own config file.

Key settings:
- `target_fps`: Frame rate (default 60)
- `enemies.enemies_per_wave`: Number of enemies per wave
- `enemies.wave_count`: Total number of waves
- `enemies.spawn_delay`: Seconds between enemy spawns
- `towers.initial_towers`: Towers placed at game start
- `towers.allow_dynamic_placement`: Enable/disable manual tower placement
- `visualization.show_*`: Toggle various visual features

## Command-Line Options

```bash
apathion play [OPTIONS]

Options:
  --algorithm TEXT       Pathfinding algorithm (astar, aco, dqn)
  --map_type TEXT       Map type (simple, branching, open_arena)
  --waves INTEGER       Number of waves
  --enemies INTEGER     Enemies per wave
  --config_file PATH    Path to JSON configuration file
```

## Examples

### Quick ACO Demo
```bash
apathion play --algorithm=aco --waves=3 --enemies=15
```

### Challenging Game
```bash
apathion play --algorithm=astar --map_type=branching --waves=10 --enemies=30
```

### Custom Configuration
```bash
apathion play --config_file=my_custom_game.json
```

## Troubleshooting

**Window doesn't appear**
- Make sure pygame is installed: `uv pip install pygame`
- Check that you have a display environment (X11, Wayland, etc.)

**Game is too slow**
- Reduce `target_fps` in config
- Lower enemy count
- Disable `show_pheromones` and `show_paths` in config

**Towers won't place**
- Make sure `allow_dynamic_placement` is true in config
- Click on walkable (non-obstacle) cells
- Avoid clicking on existing towers

**Enemies stuck**
- This shouldn't happen, but if it does, try a different pathfinding algorithm
- Check map for unreachable areas


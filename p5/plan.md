# Openmpi Integration Project Outline

1. Define function to integrate
2. Define width of trapezoid (dx)
3. distribute subdomains to workers (multiple of 2)
4. workers calculate f(x) for each dx in range, then find trapezoid area, sum areas and return result
5. Master adds results

#pragma once
#include <SDL2/SDL.h>

struct Body
{
    float mass;
    float x, y;
    float vx, vy;
    float ax, ay;
};

void renderPoints(Body* bodies, int count, SDL_Renderer* renderer)
{
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255); // set white color
    for (int i = 0; i < count; i++)
    {
        int radius = (int)(bodies[i].mass * 0.1f); // calculate radius from mass
        SDL_Rect rect = { (int)bodies[i].x - radius, (int)bodies[i].y - radius, radius * 2, radius * 2 };
        SDL_RenderFillRect(renderer, &rect); // draw a filled rectangle for each body
    }
    SDL_RenderPresent(renderer); // show the rendered points on screen
}

void updatePositions(Body* bodies, int count, float deltaTime)
{
    // update positions based on velocity and acceleration using Euler's method
    for (int i = 0; i < count; i++)
    {
        bodies[i].vx += bodies[i].ax * deltaTime;
        bodies[i].vy += bodies[i].ay * deltaTime;
        bodies[i].x += bodies[i].vx * deltaTime;
        bodies[i].y += bodies[i].vy * deltaTime;
    }
}

#include <GL/glut.h>

//---------------------------------------------------------------------
// makes a triangle for testing opengl
// install open gl and glut with 

void display() {
    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_TRIANGLES);
    glColor3f(1.0f, 0.0f, 0.0f);    // red
    glVertex2f(0.0f, 1.0f);         // top
    glColor3f(0.0f, 1.0f, 0.0f);    // green
    glVertex2f(-1.0f, -1.0f);       // bottom left
    glColor3f(0.0f, 0.0f, 1.0f);    // blue
    glVertex2f(1.0f, -1.0f);        // bottom right
    glEnd();
    glFlush();
}

void init() {
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
}

unit plotcubes;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, GL, GLU;

procedure initCubePlot;
procedure plotCube;

implementation

procedure initCubePlot;
begin

end;

procedure plotCube;
begin
  glColor3f(1.0,1.0,1.0);

   // draw a cube (6 quadrilaterals)
  glBegin(GL_QUADS);				// start drawing the cube.

	// Front Face
	glTexCoord2f(0.0, 0.0); glVertex3f(-1.0, -1.0,  1.0);	// Bottom Left Of The Texture and Quad
	glTexCoord2f(1.0, 0.0); glVertex3f( 1.0, -1.0,  1.0);	// Bottom Right Of The Texture and Quad
	glTexCoord2f(1.0, 1.0); glVertex3f( 1.0,  1.0,  1.0);	// Top Right Of The Texture and Quad
	glTexCoord2f(0.0, 1.0); glVertex3f(-1.0,  1.0,  1.0);	// Top Left Of The Texture and Quad

	// Back Face
	glTexCoord2f(1.0, 0.0); glVertex3f(-1.0, -1.0, -1.0);	// Bottom Right Of The Texture and Quad
	glTexCoord2f(1.0, 1.0); glVertex3f(-1.0,  1.0, -1.0);	// Top Right Of The Texture and Quad
	glTexCoord2f(0.0, 1.0); glVertex3f( 1.0,  1.0, -1.0);	// Top Left Of The Texture and Quad
	glTexCoord2f(0.0, 0.0); glVertex3f( 1.0, -1.0, -1.0);	// Bottom Left Of The Texture and Quad

	// Top Face
	glTexCoord2f(0.0, 1.0); glVertex3f(-1.0,  1.0, -1.0);	// Top Left Of The Texture and Quad
	glTexCoord2f(0.0, 0.0); glVertex3f(-1.0,  1.0,  1.0);	// Bottom Left Of The Texture and Quad
	glTexCoord2f(1.0, 0.0); glVertex3f( 1.0,  1.0,  1.0);	// Bottom Right Of The Texture and Quad
	glTexCoord2f(1.0, 1.0); glVertex3f( 1.0,  1.0, -1.0);	// Top Right Of The Texture and Quad

	// Bottom Face
	glTexCoord2f(1.0, 1.0); glVertex3f(-1.0, -1.0, -1.0);	// Top Right Of The Texture and Quad
	glTexCoord2f(0.0, 1.0); glVertex3f( 1.0, -1.0, -1.0);	// Top Left Of The Texture and Quad
	glTexCoord2f(0.0, 0.0); glVertex3f( 1.0, -1.0,  1.0);	// Bottom Left Of The Texture and Quad
	glTexCoord2f(1.0, 0.0); glVertex3f(-1.0, -1.0,  1.0);	// Bottom Right Of The Texture and Quad

	// Right face
	glTexCoord2f(1.0, 0.0); glVertex3f( 1.0, -1.0, -1.0);	// Bottom Right Of The Texture and Quad
	glTexCoord2f(1.0, 1.0); glVertex3f( 1.0,  1.0, -1.0);	// Top Right Of The Texture and Quad
	glTexCoord2f(0.0, 1.0); glVertex3f( 1.0,  1.0,  1.0);	// Top Left Of The Texture and Quad
	glTexCoord2f(0.0, 0.0); glVertex3f( 1.0, -1.0,  1.0);	// Bottom Left Of The Texture and Quad

	// Left Face
	glTexCoord2f(0.0, 0.0); glVertex3f(-1.0, -1.0, -1.0);	// Bottom Left Of The Texture and Quad
	glTexCoord2f(1.0, 0.0); glVertex3f(-1.0, -1.0,  1.0);	// Bottom Right Of The Texture and Quad
	glTexCoord2f(1.0, 1.0); glVertex3f(-1.0,  1.0,  1.0);	// Top Right Of The Texture and Quad
	glTexCoord2f(0.0, 1.0); glVertex3f(-1.0,  1.0, -1.0);	// Top Left Of The Texture and Quad

  glEnd();					// Done Drawing The Cube
end;


end.


unit plotcubes;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, GL, GLU, GLUT, glFonts, ImagingOpenGL, Dialogs;

procedure initCubePlot;
procedure plotCube;

implementation

var
  Tex1, Tex2: GLuint;

procedure initCubePlot;
begin
  ShowMessage('I am here');
  Load_GL_ARB_multitexture;
  glClearColor(0, 0, 0, 0);
  Tex1 := LoadGLTextureFromFile('01.png');
  Tex2 := LoadGLTextureFromFile('02.png');
  glActiveTextureARB(GL_TEXTURE0_ARB);
  glEnable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, Tex1);
  glActiveTextureARB(GL_TEXTURE1_ARB);
  glEnable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, Tex2);
  {glClearColor(0, 0, 0, 0);
  glEnable(GL_TEXTURE_2D);
  Tex1 := LoadGLTextureFromFile('01.png');
  Tex2 := LoadGLTextureFromFile('02.png');  }
end;

procedure plotCube;
begin
  glClear(GL_COLOR_BUFFER_BIT or GL_DEPTH_BUFFER_BIT);
  {
  glLoadIdentity;
  glTranslatef(0, 0, -5);}

  glBegin(GL_QUADS);
    glMultiTexCoord2fARB(GL_TEXTURE0_ARB, 1, 0);
    glMultiTexCoord2fARB(GL_TEXTURE1_ARB, 1, 0);
    glVertex3f(2.516, 2, 0);
    glMultiTexCoord2fARB(GL_TEXTURE0_ARB, 0, 0);
    glMultiTexCoord2fARB(GL_TEXTURE1_ARB, 0, 0);
    glVertex3f(-2.516, 2, 0);
    glMultiTexCoord2fARB(GL_TEXTURE0_ARB, 0, 1);
    glMultiTexCoord2fARB(GL_TEXTURE1_ARB, 0, 1);
    glVertex3f(-2.516,-2, 0);
    glMultiTexCoord2fARB(GL_TEXTURE0_ARB, 1, 1);
    glMultiTexCoord2fARB(GL_TEXTURE1_ARB, 1, 1);
    glVertex3f(2.516,-2, 0);
  glEnd;

  {
  //Tex1 := LoadGLTextureFromFile('01.png');
  glEnable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, Tex1);
  glBegin(GL_QUADS);
      glTexCoord2f(1, 0);
      glVertex3f( 1, 1, 0);
      glTexCoord2f(0, 0);
      glVertex3f(-1, 1, 0);
      glTexCoord2f(0, 1);
      glVertex3f(-1,-1, 0);
      glTexCoord2f(1, 1);
      glVertex3f( 1,-1, 0);
  glEnd;
  glDisable(GL_TEXTURE_2D);
  }
   // draw a cube (6 quadrilaterals)
  {glBegin(GL_QUADS);				// start drawing the cube.

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

  glEnd();  // Done Drawing The Cube }

  //testGLFonts();
end;


end.


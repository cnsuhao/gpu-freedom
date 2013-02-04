unit plotcubes;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, GL, GLU, GLUT, glFonts, ImagingOpenGL, Dialogs;

type TGLCube = class(TObject)
   constructor Create;
   procedure initCubePlot;
   procedure plotCube;

   private
     Tex1, Tex2, Tex3, Tex4, Tex5, Tex6: GLuint;
end;

implementation

constructor TGLCube.Create;
begin
 initCubePlot;
end;

procedure TGLCube.initCubePlot;
begin
  glClearColor(0.0, 0.0, 0.0, 0.0);
end;

procedure TGLCube.plotCube;
begin
  glClear(GL_COLOR_BUFFER_BIT or GL_DEPTH_BUFFER_BIT);
  Tex1 := LoadGLTextureFromFile('01.png');
  glEnable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, Tex1);
  glBegin(GL_QUADS);
      glTexCoord2f(1, 0);
      glVertex3f( 1, 1, -1);
      glTexCoord2f(0, 0);
      glVertex3f(-1, 1, -1);
      glTexCoord2f(0, 1);
      glVertex3f(-1,-1, -1);
      glTexCoord2f(1, 1);
      glVertex3f( 1,-1, -1);
  glEnd;

  Tex2 := LoadGLTextureFromFile('02.png');
  glBindTexture(GL_TEXTURE_2D, Tex2);
  glBegin(GL_QUADS);
      glTexCoord2f(1, 0);
      glVertex3f( 1, 1, 1);
      glTexCoord2f(0, 0);
      glVertex3f(1, 1, -1);
      glTexCoord2f(0, 1);
      glVertex3f(1,-1, -1);
      glTexCoord2f(1, 1);
      glVertex3f(1,-1, 1);
  glEnd;

  Tex3 := LoadGLTextureFromFile('03.png');
  glBindTexture(GL_TEXTURE_2D, Tex3);
  glBegin(GL_QUADS);
      glTexCoord2f(1, 0);
      glVertex3f(-1, 1, 1);
      glTexCoord2f(0, 0);
      glVertex3f(1, 1, 1);
      glTexCoord2f(0, 1);
      glVertex3f(1,-1, 1);
      glTexCoord2f(1, 1);
      glVertex3f(-1,-1, 1);
  glEnd;

  Tex4 := LoadGLTextureFromFile('04.png');
  glBindTexture(GL_TEXTURE_2D, Tex4);
  glBegin(GL_QUADS);
      glTexCoord2f(1, 0);
      glVertex3f(-1, 1, 1);
      glTexCoord2f(0, 0);
      glVertex3f(-1, 1, -1);
      glTexCoord2f(0, 1);
      glVertex3f(-1,-1, -1);
      glTexCoord2f(1, 1);
      glVertex3f(-1,-1, 1);
  glEnd;

  Tex5 := LoadGLTextureFromFile('05.png');
  glBindTexture(GL_TEXTURE_2D, Tex5);
  glBegin(GL_QUADS);
      glTexCoord2f(1, 0);
      glVertex3f(-1, 1,-1);
      glTexCoord2f(0, 0);
      glVertex3f(1, 1, -1);
      glTexCoord2f(0, 1);
      glVertex3f(1,1, 1);
      glTexCoord2f(1, 1);
      glVertex3f(-1, 1, 1);
  glEnd;

  Tex6 := LoadGLTextureFromFile('06.png');
  glBindTexture(GL_TEXTURE_2D, Tex6);
  glBegin(GL_QUADS);
      glTexCoord2f(1, 0);
      glVertex3f(-1, -1,-1);
      glTexCoord2f(0, 0);
      glVertex3f(1, -1, -1);
      glTexCoord2f(0, 1);
      glVertex3f(1, -1, 1);
      glTexCoord2f(1, 1);
      glVertex3f(-1, -1, 1);
  glEnd;


  glDisable(GL_TEXTURE_2D);
end;


end.


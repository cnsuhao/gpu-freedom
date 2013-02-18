unit glfonts;
// source from http://wiki.freepascal.org/OpenGL_Tutorial
{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, GL, GLU, GLUT;

procedure glWrite(X, Y: GLfloat; Font: Pointer; Text: String);
procedure glEnter2D;
procedure glLeave2D;

function glGetViewportWidth: Integer;
function glGetViewportHeight: Integer;

function GetTotalTime: Single;

procedure testGLFonts();

implementation

procedure glWrite(X, Y: GLfloat; Font: Pointer; Text: String);
var
  I: Integer;
begin
  glRasterPos2f(X, Y);
  for I := 1 to Length(Text) do
    glutBitmapCharacter(Font, Integer(Text[I]));
end;

function glGetViewportWidth: Integer;
var
  Rect: array[0..3] of Integer;
begin
  glGetIntegerv(GL_VIEWPORT, @Rect);
  Result := Rect[2] - Rect[0];
end;

function glGetViewportHeight: Integer;
var
  Rect: array[0..3] of Integer;
begin
  glGetIntegerv(GL_VIEWPORT, @Rect);
  Result := Rect[3] - Rect[1];
end;


procedure glEnter2D;
begin
  glMatrixMode(GL_PROJECTION);
  glPushMatrix;
  glLoadIdentity;
  gluOrtho2D(0, glGetViewportWidth, 0, glGetViewportHeight);

  glMatrixMode(GL_MODELVIEW);
  glPushMatrix;
  glLoadIdentity;

  glDisable(GL_DEPTH_TEST);
end;

procedure glLeave2D;
begin
  glMatrixMode(GL_PROJECTION);
  glPopMatrix;
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix;

  glEnable(GL_DEPTH_TEST);
end;


function GetTotalTime: Single;
begin
  Result := glutGet(GLUT_ELAPSED_TIME) / 1000;
end;


procedure testGLFonts();
begin
  glEnter2D;

  glColor3f(0.2, 0.8 + 0.2 * Sin(GetTotalTime * 5), 0);
  //glColor3f(1, 1, 1);
  glWrite(50, glGetViewportHeight - 60, GLUT_BITMAP_9_BY_15, 'GLUT_BITMAP_9_BY_15');
  glWrite(50, glGetViewportHeight - 90, GLUT_BITMAP_8_BY_13, 'GLUT_BITMAP_8_BY_13');
  glWrite(50, glGetViewportHeight - 120, GLUT_BITMAP_TIMES_ROMAN_10, 'GLUT_BITMAP_TIMES_ROMAN_10');
  glWrite(50, glGetViewportHeight - 150, GLUT_BITMAP_TIMES_ROMAN_24, 'GLUT_BITMAP_TIMES_ROMAN_24');
  glWrite(50, glGetViewportHeight - 180, GLUT_BITMAP_HELVETICA_10, 'GLUT_BITMAP_HELVETICA_10');
  glWrite(50, glGetViewportHeight - 210, GLUT_BITMAP_HELVETICA_12, 'GLUT_BITMAP_HELVETICA_12');
  glWrite(50, glGetViewportHeight - 240, GLUT_BITMAP_HELVETICA_18, 'GLUT_BITMAP_HELVETICA_18');

  glLeave2D;
end;

end.


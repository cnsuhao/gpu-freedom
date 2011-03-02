unit OpenGLSphereControl;

{$mode objfpc}{$H+}

interface

uses
  Classes, Controls, SysUtils, GL, GLU, sphere3dPlots, arcball, OpenGLContext, texturestructure;

  type
    TOpenGLSphereControl = class(TOpenGLControl)
       public
         constructor Create(obj : TComponent);
         procedure setColors(colors : PGridColor);
         procedure setRotate(rotate : Boolean);
         procedure setSphere(sphere : Boolean);

         procedure openGLSphereControlResize(Sender: TObject);
         procedure openGLSphereControlPaint(Sender: TObject);

         procedure MouseDown(Sender: TObject; Button: TMouseButton; Shift: TShiftState; X, Y: Integer);
         procedure MouseMove(Sender: TObject; Shift: TShiftState; X, Y: Integer);
         procedure MouseUp(Sender: TObject; Button: TMouseButton; Shift: TShiftState; X, Y: Integer);

       private

         theBall : TArcBall;
         cube_rotationx: GLFloat;
         cube_rotationy: GLFloat;
         cube_rotationz: GLFloat;
         _colors : PGridColor;
         _angle  : Double;
         _rotate : Boolean;
         _sphere : Boolean;
   end;

implementation

constructor TOpenGLSphereControl.Create(obj : TComponent);
begin
  inherited Create(obj);

  theBall := TArcBall.Create;
  theBall.Init(256);

  _angle := 0;
  _rotate := true;
  _sphere := true;

  onPaint := @openGLWorldControlPaint;
  onResize := @openGLWorldControlResize;
  onMouseDown := @MouseDown;
  onMouseUP := @MouseUp;
  onMouseMove := @MouseMove;
end;

procedure TOpenGLSphereControl.setColors(colors : PGridColor);
begin
  _colors := colors;
end;

procedure TOpenGLSphereControl.setRotate(rotate : Boolean);
begin
  _rotate := rotate;
end;

procedure TOpenGLSphereControl.setSphere(sphere : Boolean);
begin
  _sphere := sphere;
end;

procedure TOpenGLSphereControl.MouseDown(Sender: TObject; Button: TMouseButton;
  Shift: TShiftState; X, Y: Integer);
begin
  if Button = mbLeft then
 	begin
         theBall.BeginDrag;
 	 Paint;
 	end;
end;

procedure TOpenGLSphereControl.MouseMove(Sender: TObject; Shift: TShiftState; X,
  Y: Integer);
begin
  theBall.MouseMove(X, Y);
  Paint;
end;

procedure TOpenGLSphereControl.MouseUp(Sender: TObject; Button: TMouseButton;
  Shift: TShiftState; X, Y: Integer);
begin
  if Button = mbLeft then
 	begin
 	 theBall.EndDrag;
 	 Paint;
 	end;
end;

procedure TOpenGLSphereControl.OpenGLSphereControlPaint(Sender: TObject);
var
  sphere : PGLUquadric;
  Tex1, Tex2: GLuint;

begin
  if Sender=nil then ;

  glClearColor(0.0, 0.0, 0.0, 0.0);
  glClear(GL_COLOR_BUFFER_BIT or GL_DEPTH_BUFFER_BIT);
  glEnable(GL_DEPTH_TEST);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(-2, 2, -1.5, 1.5, -2, 2);
  glMatrixMode(GL_MODELVIEW);

  glLoadIdentity();
  glLoadMatrixf(@theBall.Matrix);
  if _rotate then glRotatef(_angle, 0.0, 1.0, 0.0);

  glEnable(GL_BLEND);
  if _sphere then
    plot3dSphere(_colors)
  else
    plot3dTerrain(_colors);
  glDisable(GL_BLEND);


  SwapBuffers;
  if _rotate then
     begin
       _angle := _angle + 1;
       if (_angle>359) then _angle := 0;
     end;
end;

procedure TOpenGLSphereControl.openGLSphereControlResize(Sender: TObject);
begin
  if Sender=nil then ;
  if Height <= 0 then exit;
end;


end.


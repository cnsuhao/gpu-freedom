unit cubeUnit;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, FileUtil, Forms, Controls, Graphics, Dialogs,
  OpenGLCubeControl;

type

  { TfrmCube }

  TfrmCube = class(TForm)
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
    procedure FormPaint(Sender: TObject);
  private
  public
    OpenGLCubeControl : TOpenGLCubeControl;
    procedure OnAppIdle(Sender: TObject; var Done: Boolean);
  end;

var
  frmCube: TfrmCube;

implementation

{$R *.lfm}

{ TfrmCube }

procedure TfrmCube.FormCreate(Sender: TObject);
begin
  OpenGLCubeControl:=TOpenGLCubeControl.Create(Self);
    with OpenGLCubeControl do begin
      Name:='OpenGLCubeControl';
      Align:=alNone;
      Parent:=Self;
      Top := 0;
      Left := 0;
      Height := frmCube.Height - Top - 15;
      Width := 2 * 360 + 15;
    end;
    OpenGLCubeControl.setRotate(false);
    Application.AddOnIdleHandler(@OnAppIdle);
end;

procedure TfrmCube.FormDestroy(Sender: TObject);
begin
  OpenGLCubeControl.Free;
end;

procedure TfrmCube.FormPaint(Sender: TObject);
begin
 OpenGlCubeControl.AutoResizeViewport := true;

 if frmCube.Height>640 then
  OpenGLCubeControl.Height := frmCube.Height - OpenGLCubeControl.Top - 15;
 if frmCube.Width > 2*360  + 15 + 210 then
    OpenGLCubeControl.Width := frmCube.Width - 210;

 {
 gbLine.Left := OpenGLCubeControl.Width+15;
 gbStatus.Left := OpenGLCubeControl.Width+15;
 gbTime.Left := OpenGLCubeControl.Width+15;
 }
 OpenGlCubeControl.AutoResizeViewport := false;
end;

procedure TfrmCube.OnAppIdle(Sender: TObject; var Done: Boolean);
begin
  Done:=false;
  OpenGLCubeControl.Invalidate;
end;

end.


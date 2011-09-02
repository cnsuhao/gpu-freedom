unit netmapperforms;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, FileUtil, LResources, Forms, Controls, Graphics, Dialogs,
  openglspherecontrol, earthtextures, texturestructure;

type

  { TNetmapperForm }

  TNetmapperForm = class(TForm)
     openGLSphereControl : TOpenGLSphereControl;

     procedure FormCreate(Sender: TObject);
     procedure FormDestroy(Sender: TObject);
     procedure FormHide(Sender: TObject);
     procedure FormShow(Sender: TObject);
     procedure OnAppIdle(Sender: TObject; var Done: Boolean);

    private
     earthTexturer_ : TEarthTexturer;
  end; 

var
  NetmapperForm: TNetmapperForm;

implementation

procedure TNetmapperForm.FormCreate(Sender: TObject);
begin
  earthTexturer_ := TEarthTexturer.Create();

  openGLSphereControl := TOpenGLSphereControl.Create(self);
  with OpenGLSphereControl do begin
    Name:='OpenGLSphereControl';
    Align:=alNone;
    Parent:=Self;
    Top := 10;
    Left := 10;
    Height := 360;
    Width := 480;
    Visible := true;
  end;


  openGLSphereControl.setColors(earthTexturer_.getTexture());

  Application.AddOnIdleHandler(@OnAppIdle);
end;

procedure TNetmapperForm.FormDestroy(Sender: TObject);
begin
  earthTexturer_.Free;
  openGLSphereControl.Free;
end;

procedure TNetmapperForm.FormHide(Sender: TObject);
begin
//
end;

procedure TNetmapperForm.FormShow(Sender: TObject);
begin
//
end;

procedure TNetmapperForm.OnAppIdle(Sender: TObject; var Done: Boolean);
begin
if Visible then
  begin
     Done:=false;
     openGLSphereControl.Invalidate;
  end;
end;

initialization
  {$I netmapperforms.lrs}

end.


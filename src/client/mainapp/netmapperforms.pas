unit netmapperforms;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, FileUtil, LResources, Forms, Controls, Graphics, Dialogs,
  StdCtrls, ExtCtrls, openglspherecontrol, earthtextures, texturestructure;

const
  BORDER = 10;

type

  { TNetmapperForm }

  TNetmapperForm = class(TForm)
      cbRotate: TCheckBox;
     openGLSphereControl : TOpenGLSphereControl;
     pnlTop: TPanel;

     procedure cbRotateChange(Sender: TObject);
     procedure FormCreate(Sender: TObject);
     procedure FormDestroy(Sender: TObject);
     procedure FormHide(Sender: TObject);
     procedure FormResize(Sender: TObject);
     procedure FormShow(Sender: TObject);
     procedure OnAppIdle(Sender: TObject; var Done: Boolean);

    private
     earthTexturer_ : TEarthTexturer;

    procedure resizeWindow;
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
    Visible := true;
  end;
  resizeWindow;
  cbRotate.Checked := false;

  openGLSphereControl.setColors(earthTexturer_.getTexture());
  Application.AddOnIdleHandler(@OnAppIdle);
end;

procedure TNetmapperForm.cbRotateChange(Sender: TObject);
begin
  OpenGLSphereControl.setRotate(cbRotate.Checked);
end;

procedure TNetmapperForm.resizeWindow;
begin
 pnlTop.Width := self.Width;

 with OpenGLSphereControl do begin
  Top := pnlTop.Height+BORDER;
  Left := BORDER;
  Height := self.Height-pnlTop.Height - (BORDER * 2);
  Width := self.Width - (BORDER * 2);
 end;
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

procedure TNetmapperForm.FormResize(Sender: TObject);
begin
 resizeWindow;
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
Application.ProcessMessages;
end;

initialization
  {$I netmapperforms.lrs}

end.


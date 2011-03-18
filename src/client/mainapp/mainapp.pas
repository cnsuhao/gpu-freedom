unit mainapp;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, FileUtil, LResources, Forms, Controls, Graphics, Dialogs,
  coreobjects;

type

  { TGPUMainApp }

  TGPUMainApp = class(TForm)
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
  end;

var
  GPUMainApp: TGPUMainApp;

implementation

{ TGPUMainApp }

procedure TGPUMainApp.FormCreate(Sender: TObject);
begin

end;

procedure TGPUMainApp.FormDestroy(Sender: TObject);
begin

end;



initialization
  {$I mainapp.lrs}

end.



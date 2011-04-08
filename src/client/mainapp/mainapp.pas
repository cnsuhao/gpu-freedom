unit mainapp;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, FileUtil, LResources, Forms, Controls, Graphics, Dialogs,
  coreobjects, ExtCtrls;

type

  { TGPUMainApp }

  TGPUMainApp = class(TForm)
    MainTimer: TTimer;
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
    procedure MainTimerTimer(Sender: TObject);
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

procedure TGPUMainApp.MainTimerTimer(Sender: TObject);
begin
  if serviceman <> nil then serviceman.clearFinishedThreads;
end;



initialization
  {$I mainapp.lrs}

end.



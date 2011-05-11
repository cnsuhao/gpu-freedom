unit mainapp;

{$mode objfpc}{$H+}
{$DEFINE COREINCLUDED}

interface

uses
  Classes, SysUtils, FileUtil, LResources, Forms, Controls, Graphics, Dialogs,
  coreobjects, coreloops, ExtCtrls;

type

  { TGPUMainApp }

  TGPUMainApp = class(TForm)
    MainTimer: TTimer;
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
    procedure MainTimerTimer(Sender: TObject);
   private
    coreloop_ : TCoreLoop;
  end;

var
  GPUMainApp: TGPUMainApp;

implementation

{ TGPUMainApp }

procedure TGPUMainApp.FormCreate(Sender: TObject);
begin
  {$IFDEF COREINCLUDED}
   coreloop_ := TCoreLoop.Create();
   coreloop_.start;
  {$ENDIF}
end;

procedure TGPUMainApp.FormDestroy(Sender: TObject);
begin
  {$IFDEF COREINCLUDED}
   coreloop_.stop();
   coreloop_.Free;
  {$ENDIF}
end;

procedure TGPUMainApp.MainTimerTimer(Sender: TObject);
begin
  if serviceman <> nil then serviceman.clearFinishedThreads;
  {$IFDEF COREINCLUDED}
  coreloop_.tick;
  {$ENDIF}
end;



initialization
  {$I mainapp.lrs}

end.



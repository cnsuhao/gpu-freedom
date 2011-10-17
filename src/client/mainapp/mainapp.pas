unit mainapp;

{$mode objfpc}{$H+}
{$DEFINE COREINCLUDED}

interface

uses
  Classes, SysUtils, FileUtil, LResources, Forms, Controls, Graphics, Dialogs,
  coreobjects, coreloops, ExtCtrls, Buttons, chatforms, netmapperforms,
  parametersforms, StdCtrls, Process, LazHelpHTML;

type

  { TGPUMainApp }

  TGPUMainApp = class(TForm)
    btnSQL: TButton;
    btnWeb: TButton;
    MainTimer: TTimer;
    sbtnChat: TSpeedButton;
    sbtnConfig: TSpeedButton;
    sbtnNetmapper: TSpeedButton;
    trayIcon: TTrayIcon;
    procedure btnSQLClick(Sender: TObject);
    procedure btnWebClick(Sender: TObject);
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
    procedure MainTimerTimer(Sender: TObject);
    procedure sbtnChatClick(Sender: TObject);
    procedure sbtnConfigClick(Sender: TObject);
    procedure sbtnNetmapperClick(Sender: TObject);
   private
    coreloop_ : TCoreLoop;
    appPath_  : String;
    BrowserPath_,
    BrowserParams_: string;
  end;

var
  GPUMainApp: TGPUMainApp;

implementation

{ TGPUMainApp }

procedure TGPUMainApp.FormCreate(Sender: TObject);
var v : THTMLBrowserHelpViewer;
begin
  {$IFDEF COREINCLUDED}
   coreloop_ := TCoreLoop.Create();
   coreloop_.start;
  {$ENDIF}
  appPath_ := ExtractFilePath(ParamStr(0));

 try
    v:=THTMLBrowserHelpViewer.Create(nil);
    v.FindDefaultBrowser(BrowserPath_,BrowserParams_);
 finally
    v.Free;
 end;
end;

procedure TGPUMainApp.btnSQLClick(Sender: TObject);
var aProcess : TProcess;
begin
  AProcess := TProcess.Create(nil);
  try
    AProcess.CommandLine := '"'+appPath_+PathDelim+'sqlite3.exe" "'+appPath_+PathDelim+'coreapp.db'+'"';
    AProcess.Options := AProcess.Options - [poWaitOnExit];
    AProcess.Execute;
  finally
    AProcess.Free;
  end;
end;

procedure TGPUMainApp.btnWebClick(Sender: TObject);
var BrowserProcess: TProcess;
    p             : Longint;
    params        : String;
begin
    params := BrowserParams_;
    p:=System.Pos('%s', params);
    System.Delete(params,p,2);
    System.Insert('http://gpu.sourceforge.net',params,p);
    BrowserProcess:=TProcess.Create(nil);
    try
      BrowserProcess.CommandLine:='"'+BrowserPath_+'" '+params;
      BrowserProcess.Options := BrowserProcess.Options - [poWaitOnExit];
      BrowserProcess.Execute;
    finally
      BrowserProcess.Free;
    end;
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

procedure TGPUMainApp.sbtnChatClick(Sender: TObject);
begin
  chatForm.Visible := not chatForm.Visible;
end;

procedure TGPUMainApp.sbtnConfigClick(Sender: TObject);
begin
  parametersForm.Visible := not parametersForm.Visible;
end;

procedure TGPUMainApp.sbtnNetmapperClick(Sender: TObject);
begin
  netmapperForm.Visible := not netMapperForm.Visible;
end;



initialization
  {$I mainapp.lrs}

end.



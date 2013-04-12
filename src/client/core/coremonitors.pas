unit coremonitors;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, lockfiles, Process;

var coreprocess : TProcess;

type TCoreMonitor = class(TObject)
  public
    constructor Create(corenumber : Longint);
    destructor Destroy();

    // used in core
    procedure createLockFile;
    function  coreCanRun : Boolean;
    function  coreCanStart : Boolean;
    procedure removeLockFile;

    // called by GUI
    procedure startCore;
    procedure stopCore;
    function  isCoreRunning() : Boolean;

  private
    path_          : String;
    isrunninglock_ : TLockFile;
end;

implementation

constructor TCoreMonitor.Create(corenumber : Longint);
begin
   inherited Create;

   path_ := extractFilePath(ParamStr(0));
   isrunninglock_     := TLockFile.Create(path_+'locks', 'gpucore_'+IntToStr(corenumber)+'.lock');
end;

destructor  TCoreMonitor.Destroy();
begin
   isrunninglock_.Free;

   inherited Destroy;
end;

procedure   TCoreMonitor.createLockFile;
begin
  isrunninglock_.createLF;
end;

procedure TCoreMonitor.removeLockFile;
begin
 isrunninglock_.delete;
end;

function    TCoreMonitor.coreCanRun : Boolean;
begin
  Result := isrunninglock_.exists;
end;

function    TCoreMonitor.coreCanStart : Boolean;
begin
  Result := not isrunninglock_.exists;
end;


procedure   TCoreMonitor.startCore;
var path : String;
begin
 coreprocess := TProcess.Create(nil);
 coreprocess.CommandLine := path+'gpucore.exe';
 coreprocess.Options := coreprocess.Options; // e.g.  +[poWaitOnExit];
 coreprocess.Execute;
end;

procedure   TCoreMonitor.stopCore;
begin
 isrunninglock_.delete;
 sleep(2000);
 if coreprocess<>nil then coreprocess.Free;
end;

function    TCoreMonitor.isCoreRunning() : Boolean;
begin
  Result := isrunninglock_.exists;
end;

end.


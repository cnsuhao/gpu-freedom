unit coremonitors;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, lockfiles;

type TCoreMonitor = class(TObject)
  public
    constructor Create();
    destructor Destroy();

    // used in core
    procedure coreStarted;
    function  coreCanRun : Boolean;
    procedure coreStopped;

    // called by GUI
    procedure startCore;
    procedure stopCore;
    function  isCoreRunning() : Boolean;

  private
    path_          : String;
    isrunninglock_ : TLockFile;
end;

implementation

constructor TCoreMonitor.Create();
begin
   path_ := extractFilePath(ParamStr(0));
   isrunninglock_     := TLockFile.Create(path_+PathDelim+'locks', 'coreapp.lock');
end;

destructor  TCoreMonitor.Destroy();
begin
   isrunninglock_.Free;
end;

procedure   TCoreMonitor.coreStarted;
begin
  isrunninglock_.createLF;
end;

procedure TCoreMonitor.coreStopped;
begin
 isrunninglock_.delete;
end;

function    TCoreMonitor.coreCanRun : Boolean;
begin
  Result := isrunninglock_.exists;
end;

procedure   TCoreMonitor.startCore;
begin
 //TODO: implement this function
end;

procedure   TCoreMonitor.stopCore;
begin
 isrunninglock_.delete;
end;

function    TCoreMonitor.isCoreRunning() : Boolean;
begin
  Result := isrunninglock_.exists;
end;

end.


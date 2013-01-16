program parliasim;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}{$IFDEF UseCThreads}
  cthreads,
  {$ENDIF}{$ENDIF}
  Classes, SysUtils, CustApp, parliamentstructure, parliamentoutput, simulation
  { you can add units after this };

type

  { TParliamentSim }

  TParliamentSim = class(TCustomApplication)
  protected
    procedure DoRun; override;
  public
    constructor Create(TheOwner: TComponent); override;
    destructor Destroy; override;
  end;

{ TParliamentSim }

procedure TParliamentSim.DoRun;
var
  ErrorMsg: String;
  par     : TParliament;
begin
  initParliament(par, 300,2,0.55);
  printParliament(par);
  simulateParliament(par);
  WriteLn('Parliament simulation finished');
  Readln;
  Terminate;
end;

constructor TParliamentSim.Create(TheOwner: TComponent);
begin
  inherited Create(TheOwner);
end;

destructor TParliamentSim.Destroy;
begin
  inherited Destroy;
end;

var
  Application: TParliamentSim;
begin
  Application:=TParliamentSim.Create(nil);
  Application.Title:='ParliamentSim';
  Application.Run;
  Application.Free;
end.


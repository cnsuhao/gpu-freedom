program parliasim;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}{$IFDEF UseCThreads}
  cthreads,
  {$ENDIF}{$ENDIF}
  Classes, SysUtils, CustApp, parliamentstructure, parliamentoutput, simulation,
  statistics
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
var   par      : TParliament;
      laws     : TLaws;
      simstats : TSimStats;


procedure TParliamentSim.DoRun;
var
  ErrorMsg: String;
begin
  initParliament(par, 300,2,0.55);
  printParliament(par);
  initLaws(laws, 3000);
  initSimStats(simstats, 1000);
  simulateParliament(par, laws);
  collectStatistics(par, laws, simstats, 1);
  printStatistic(simstats, 1);
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


program fpcunittests;

{$mode objfpc}{$H+}

uses
  Classes, consoletestrunner,
  teststacks, testplugins, testpluginmanagers, testargretrievers, coremodules;

type

  { TLazTestRunner }

  TMyTestRunner = class(TTestRunner)
  protected
  // override the protected methods of TTestRunner to customize its behavior
  end;

var
  Application: TMyTestRunner;

{$IFDEF WINDOWS}{$R fpcunittests.rc}{$ENDIF}

begin
  Application := TMyTestRunner.Create(nil);
  Application.Initialize;
  Application.Title := 'FPCUnit Console test runner';
  Application.Run;
  Application.Free;
end.

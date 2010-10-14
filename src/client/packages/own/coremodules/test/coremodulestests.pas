program coremodulestests;

{$mode objfpc}{$H+}

uses
  Classes, consoletestrunner,
  teststacks, testplugins, testpluginmanagers, testargretrievers,
  testmethodcontrollers, testresultcollectors, testfrontendmanagers,
  testjobparsers, coremodules,
  testspecialcommands, testcomputationthreads, testthreadmanagers;

type

  { TLazTestRunner }

  TMyTestRunner = class(TTestRunner)
  protected
  // override the protected methods of TTestRunner to customize its behavior
  end;

var
  Application: TMyTestRunner;

{$IFDEF WINDOWS}{$R coremodulestests.rc}{$ENDIF}

begin
  Application := TMyTestRunner.Create(nil);
  Application.Initialize;
  Application.Title := 'FPCUnit Console test runner for Core Modules';
  Application.Run;
  Application.Free;
end.

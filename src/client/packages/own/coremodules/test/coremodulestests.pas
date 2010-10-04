program coremodulestests;

{$mode objfpc}{$H+}

uses
  Classes, consoletestrunner,
  teststacks, testplugins, testpluginmanagers, testargretrievers,
  testmethodcontrollers, testresultcollectors, testfrontendmanagers,
  jobparsers, threadmanagers, computationthreads, coremodules,
testspecialcommands;

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
  Application.Title := 'FPCUnit Console test runner';
  Application.Run;
  Application.Free;
end.

unit testcompthreadmanagers;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testutils, testregistry,
  computationthreads, pluginmanagers, methodcontrollers,
  loggers, resultcollectors, frontendmanagers, specialcommands,
  jobs, stacks, compthreadmanagers;

type

  TTestThreadManager= class(TTestCase)
  protected
    procedure SetUp; override; 
    procedure TearDown; override; 
  published
    procedure TestThreadManager;
  private
    plugman_   : TPluginManager;
    meth_      : TMethodController;
    logger_    : TLogger;
    res_       : TResultCollector;
    frontman_  : TFrontendManager;
    job1_,
    job2_,
    job3_,
    job4_,
    job5_      : TJob;
    threadman_ : TCompThreadManager;
  end; 

implementation

procedure TTestThreadManager.TestThreadManager;
begin
  AssertEquals('Threadman is idle', true, threadman_.isIdle);
  threadman_.setMaxThreads(1);
  AssertEquals('Maximum of threads is', 1, threadman_.getMaxThreads());
  job1_.Job := '720000, montecarlo_pi';
  threadman_.Compute(job1_);
  AssertEquals('Threadman is not idle', false, threadman_.isIdle);
  AssertEquals('Threadman does not have resources', false, threadman_.hasResources());
  threadman_.setMaxThreads(2);
  AssertEquals('Threadman is not idle', false, threadman_.isIdle);
  AssertEquals('Threadman has resources', true, threadman_.hasResources());
  job2_.Job := '720000, montecarlo_pi';
  threadman_.Compute(job2_);
  AssertEquals('Threadman is not idle', false, threadman_.isIdle);
  AssertEquals('Threadman does not have resources', false, threadman_.hasResources());

  while not threadman_.isIdle do
    begin
      Sleep(1000);
      threadman_.ClearFinishedThreads();
    end;

  AssertEquals('Threadman is idle', true, threadman_.isIdle);
  AssertEquals('Threadman has resources', true, threadman_.hasResources());

  threadman_.setMaxThreads(4);
  job1_.clear();
  job2_.clear();
  job1_.Job := '720000, random_walk';
  job2_.Job := '720000, random_walk';
  job3_.Job := '720000, random_walk';
  threadman_.Compute(job1_);
  threadman_.Compute(job2_);
  threadman_.Compute(job3_);
  AssertEquals('Threadman is idle', false, threadman_.isIdle);
  AssertEquals('Threadman has resources', true, threadman_.hasResources());
  threadman_.setMaxThreads(1);
  AssertEquals('Threadman has resources', false, threadman_.hasResources());

    while not threadman_.isIdle do
    begin
      Sleep(1000);
      threadman_.ClearFinishedThreads();
    end;

  AssertEquals('Threadman is idle', true, threadman_.isIdle);
  AssertEquals('Threadman has resources', true, threadman_.hasResources());
end;

procedure TTestThreadManager.SetUp; 
var path, extension : String;
    error           : TStkError;
begin
 clearError(error);
 path            := ExtractFilePath(ParamStr(0));
 extension       := 'dll';
 logger_         := TLogger.Create(path+PathDelim+'logs', 'core.log');
 plugman_        := TPluginManager.Create(path+PathDelim+'plugins'+PathDelim+'lib', extension, logger_);
 plugman_.loadAll(error);
 meth_           := TMethodController.Create();
 res_            := TResultCollector.Create();
 frontman_       := TFrontendManager.Create();
 job1_           := TJob.Create();
 job2_           := TJob.Create();
 job3_           := TJob.Create();
 job4_           := TJob.Create();
 job5_           := TJob.Create();
 threadman_      := TCompThreadManager.Create(plugman_, meth_, res_, frontman_);
end;

procedure TTestThreadManager.TearDown; 
begin
 frontman_.Free;
 res_.Free;
 meth_.Free;
 plugman_.Free;
 logger_.Free;
 job1_.Free;
 job2_.Free;
 job3_.Free;
 job4_.Free;
 job5_.Free;
 threadman_.Free;
end; 

initialization

  RegisterTest(TTestThreadManager); 
end.


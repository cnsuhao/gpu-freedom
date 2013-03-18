unit workflowancestors;
{
  This class is the ancestor for all workflows.
  Tables which have a status column typically implement a child of
  TWorkflowAncestor.

  (c) 2013 HB9TVM and the Global Processing Unit Team

}
interface

uses SyncObjs,
     dbtablemanagers, loggers;

type TWorkflowAncestor = class(TObject)
      constructor Create(var tableman : TDbTableManager; var logger : TLogger);
      destructor  Destroy;

    protected
      CS_       : TCriticalSection;
      logger_   : TLogger;
      tableman_ : TDbTableManager;
end;

implementation

constructor TWorkflowAncestor.Create(var tableman : TDbTableManager; var logger : TLogger);
begin
  inherited Create;
  CS_ := TCriticalSection.Create;
  logger_ := logger;
  tableman_ := tableman;
end;

destructor TWorkflowAncestor.Destroy;
begin
  if Assigned(CS_) then CS_.Free;
  inherited Destroy;
end;

end.

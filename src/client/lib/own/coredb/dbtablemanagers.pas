unit dbtablemanagers;
{
   TDbTableManager handles all tables in the sqlite database
    specified in the constructor.dbtablemanagers

   (c) by 2010-2013 HB9TVM and the GPU Team
}

interface

uses clienttables, servertables, channeltables, retrievedtables,
     jobdefinitiontables, jobqueuetables, jobresulttables, parametertables,
     jobstatstables;


type TDbTableManager = class(TObject)
  public
    constructor Create(filename : String);
    destructor Destroy;

    procedure openAll();
    procedure closeAll();

    function getClientTable() : TDbClientTable;
    function getServerTable() : TDbServerTable;
    function getChannelTable() : TDbChannelTable;
    function getRetrievedTable() : TDbRetrievedTable;

    function getJobDefinitionTable() : TDbJobDefinitionTable;
    function getJobResultTable() : TDbJobResultTable;
    function getJobQueueTable() : TDbJobQueueTable;
    function getJobStatsTable() : TDbJobStatsTable;

    function getParameterTable() : TDbParameterTable;


  private
    clienttable_    : TDbClientTable;
    servertable_    : TDbServerTable;
    chantable_      : TDbChannelTable;
    retrtable_      : TDbRetrievedTable;
    jobdefinitiontable_   : TDbJobDefinitionTable;
    jobresulttable_ : TDbJobResultTable;
    jobqueuetable_  : TDbJobQueueTable;
    jobstatstable_  : TDbJobStatsTable;
    parametertable_ : TDbParameterTable;
end;

implementation

constructor TDbTableManager.Create(filename : String);
begin
  servertable_ := TDbServerTable.Create(filename);
  clienttable_ := TDbClientTable.Create(filename);
  chantable_   := TDbChannelTable.Create(filename);
  retrtable_   := TDbRetrievedTable.Create(filename);
  jobdefinitiontable_ := TDbJobDefinitionTable.Create(filename);
  jobresulttable_ := TDbJobResultTable.Create(filename);
  jobqueuetable_  := TDbJobQueueTable.Create(filename);
  jobstatstable_  := TDbJobStatsTable.Create(filename);
  parametertable_ := TDbParameterTable.Create(filename);
end;


destructor TDbTableManager.Destroy;
begin
 clienttable_.Free;
 servertable_.Free;
 chantable_.Free;
 retrtable_.Free;
 jobdefinitiontable_.Free;
 jobresulttable_.Free;
 jobqueuetable_.Free;
 jobstatstable_.Free;
 parametertable_.Free;
end;

procedure TDbTableManager.openAll();
begin
  clienttable_.Open;
  servertable_.Open;
  chantable_.Open;
  retrtable_.Open;
  jobdefinitiontable_.Open;
  jobresulttable_.Open;
  jobqueuetable_.Open;
  jobstatstable_.Open;
  parametertable_.Open;
end;

procedure TDbTableManager.closeAll();
begin
  clienttable_.Close;
  servertable_.Close;
  chantable_.Close;
  retrtable_.Close;
  jobdefinitiontable_.Close;
  jobresulttable_.Close;
  jobqueuetable_.Close;
  jobstatstable_.Close;
  parametertable_.Close;
end;

function TDbTableManager.getClientTable() : TDbClientTable;
begin
  Result := clienttable_;
end;

function TDbTableManager.getServerTable() : TDbServerTable;
begin
  Result := servertable_;
end;

function TDbTableManager.getChannelTable() : TDbChannelTable;
begin
  Result := chantable_;
end;

function TDbTableManager.getRetrievedTable() : TDbRetrievedTable;
begin
  Result := retrtable_;
end;

function TDbTableManager.getJobDefinitionTable() : TDbJobDefinitionTable;
begin
  Result := jobdefinitiontable_;
end;

function TDbTableManager.getJobResultTable() : TDbJobResultTable;
begin
  Result := jobresulttable_;
end;

function TDbTableManager.getJobQueueTable() : TDbJobQueueTable;
begin
  Result := jobqueuetable_;
end;

function TDbTableManager.getJobStatsTable() : TDbJobStatsTable;
begin
  Result := jobstatstable_;
end;


function TDbTableManager.getParameterTable() : TDbParameterTable;
begin
  Result := parametertable_;
end;

end.

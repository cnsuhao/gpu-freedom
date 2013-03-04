unit dbtablemanagers;
{
   TDbTableManager handles all tables in the sqlite database
    specified in the constructor.dbtablemanagers

   (c) by 2010 HB9TVM and the GPU Team
}

interface

uses clienttables, servertables, channeltables, retrievedtables,
     jobtables, jobqueuetables, jobresulttables;


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

    function getJobTable() : TDbJobTable;
    function getJobResultTable() : TDbJobResultTable;
    function getJobQueueTable() : TDbJobQueueTable;

  private
    clienttable_    : TDbClientTable;
    servertable_    : TDbServerTable;
    chantable_      : TDbChannelTable;
    retrtable_      : TDbRetrievedTable;
    jobtable_       : TDbJobTable;
    jobresulttable_ : TDbJobResultTable;
    jobqueuetable_  : TDbJobQueueTable;
end;

implementation

constructor TDbTableManager.Create(filename : String);
begin
  servertable_ := TDbServerTable.Create(filename);
  clienttable_ := TDbClientTable.Create(filename);
  chantable_   := TDbChannelTable.Create(filename);
  retrtable_   := TDbRetrievedTable.Create(filename);
  jobtable_       := TDbJobTable.Create(filename);
  jobresulttable_ := TDbJobResultTable.Create(filename);
  jobqueuetable_  := TDbJobQueueTable.Create(filename);
end;


destructor TDbTableManager.Destroy;
begin
 clienttable_.Free;
 servertable_.Free;
 chantable_.Free;
 retrtable_.Free;
 jobtable_.Free;
 jobresulttable_.Free;
 jobqueuetable_.Free;
end;

procedure TDbTableManager.openAll();
begin
  clienttable_.Open;
  servertable_.Open;
  chantable_.Open;
  retrtable_.Open;
  jobtable_.Open;
  jobresulttable_.Open;
  jobqueuetable_.Open;
end;

procedure TDbTableManager.closeAll();
begin
  clienttable_.Close;
  servertable_.Close;
  chantable_.Close;
  retrtable_.Close;
  jobtable_.Close;
  jobresulttable_.Close;
  jobqueuetable_.Close;
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

function TDbTableManager.getJobTable() : TDbJobTable;
begin
  Result := jobtable_;
end;

function TDbTableManager.getJobResultTable() : TDbJobResultTable;
begin
  Result := jobresulttable_;
end;

function TDbTableManager.getJobQueueTable() : TDbJobQueueTable;
begin
  Result := jobqueuetable_;
end;

end.

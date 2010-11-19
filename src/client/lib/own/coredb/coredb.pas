unit coredb;
{
  TCoreDb represents the core sqlite database and can
   be seen as manager to retrieve tables inside the database.coredb

}
interface

uses nodetable;

type TCoreDb = object
   public
    constructor Create(filename : String);
    destructor Destroy;

    function getNodeTable : TDbNodeTable;

   private
    nodetable_ : TDbNodeTable;
end;

implementation

constructor Create(filename : String);
begin
  inherited Create;

  nodetable_ := TDbNodeTable.Create(filename);
end;

destructor Destroy;
begin

  nodetable_.Free;
  inherited Destroy;
end;

function getNodeTable : TDbNodeTable;
begin
  Result := nodetable_;
end;


end;

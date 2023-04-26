import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import org.apache.jena.graph.Node;
import org.apache.jena.graph.Triple;
import java.io.IOException;
import java.net.*;
import java.net.http.HttpResponse;
import java.util.ArrayList;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;

public class NaviUpdateOrganizer {

    protected ArrayList<Triple> added_triples;
    protected ArrayList<Triple> deleted_triples;

    public NaviUpdateOrganizer(){
        added_triples  = new ArrayList();
        deleted_triples  = new ArrayList();
    }

    public void clear(){
        added_triples.clear();
        deleted_triples.clear();
    }

    public void appendTriple(Triple triple, String operation){

        if(operation.equals("add")){
            added_triples.add(triple);
        } else if (operation.equals("delete")) {
            deleted_triples.add(triple);
        }
    }

    public JsonObject node2json(Node node){

        JsonObject tmp = new JsonObject();

        if (node.isURI()){
            tmp.addProperty("type", "uri");
            tmp.addProperty("value", node.getURI());
        } else if (node.isBlank()) {
            tmp.addProperty("type", "uri");
            tmp.addProperty("value", "_:" + node.getBlankNodeId().toString());
        } else if (node.isLiteral()) {
            tmp.addProperty("type", "literal");
            tmp.addProperty("datatype", node.getLiteralDatatypeURI());
            tmp.addProperty("value", node.getLiteralLexicalForm());
        }

        return tmp;
    }

    public JsonArray list_to_array(String operation) {

        ArrayList<Triple> list;

        if(operation.equals("add")){
            list = added_triples;
        } else if (operation.equals("delete")) {
            list = deleted_triples;
        }else{
            list = new ArrayList();
        }

        JsonArray array = new JsonArray();

        for(Triple triple : list) {
            JsonObject item = new JsonObject();
            item.add("s", node2json(triple.getSubject()));
            item.add("p", node2json(triple.getPredicate()));
            item.add("o", node2json(triple.getObject()));
            array.add(item);
        }

        return array;

    }


    public void commit() throws IOException, URISyntaxException, InterruptedException {

        JsonObject full = new JsonObject();

        full.add("delete", list_to_array("delete"));
        full.add("add", list_to_array("add"));

        String requestBody = full.toString();

        HttpClient client = HttpClient.newHttpClient();
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create("http://localhost:5000/update"))
                .version(HttpClient.Version.HTTP_1_1)
                .POST(HttpRequest.BodyPublishers.ofString(requestBody))
                .build();

        HttpResponse<String> response = client.send(request,
                HttpResponse.BodyHandlers.ofString());

//        System.out.println(response.body());
    }

    public static void main(String[] args) throws InterruptedException {

        NaviUpdateOrganizer organizer = new NaviUpdateOrganizer();
        try {
            organizer.commit();
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (URISyntaxException e) {
            throw new RuntimeException(e);
        }
    }


}

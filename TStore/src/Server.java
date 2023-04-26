/*
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 */

import org.apache.jena.fuseki.main.FusekiServer;
import org.apache.jena.rdf.model.Model;
import org.apache.jena.rdf.model.ModelFactory;
import org.apache.jena.sparql.core.DatasetGraph ;
import org.apache.jena.sparql.core.DatasetGraphFactory ;
import org.seaborne.delta.lib.LogX;
import org.seaborne.patch.RDFPatchOps ;
import org.seaborne.patch.text.RDFChangesWriterText;

public class Server {
    static { LogX.setJavaLogging(); }

    public static void main(String ...args) {
        int PORT = 2020 ;
        // In-memory dataset
        DatasetGraph dsgBase = DatasetGraphFactory.createTxnMem();
        Model model = ModelFactory.createDefaultModel() ;
        model.read("aifb_stripped.nt") ;
        dsgBase.setDefaultGraph(model.getGraph());

        try (RDFChangesWriterText changeLog = NaviWriter.create(System.out)) {

            DatasetGraph dsg = RDFPatchOps.changes(dsgBase, changeLog);
            FusekiServer server =  FusekiServer.create()
                    .port(PORT)
                    .add("/ds", dsg)
                    .build();
            server.start();

            server.join();

        }catch(Exception e){

            System.out.println(e);
            e.printStackTrace();
        }
    }
}

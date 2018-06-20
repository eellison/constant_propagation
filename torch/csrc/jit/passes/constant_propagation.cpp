#include "torch/csrc/jit/passes/constant_propagation.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/utils/functional.h"
#include "torch/csrc/jit/interpreter.h"

namespace torch { namespace jit {

at::ArrayRef<torch::jit::Value *> runNode(Node *n) {
  std::cout << " a0 \n";
  auto graph = std::make_shared<Graph>();
  auto block = graph->block();

  std::unordered_map<Value*, Value*> local_map;
  auto env = [&](Value * v) {
    auto it = local_map.find(v);
    if(it != local_map.end())
      return it->second;
    barf("Encountered a use of a value not in scope");
  };
  std::cout << " a1 \n";
  for(auto input : n->inputs()) {
    local_map[input] = block->addInput()->copyMetadata(input)->setStage(input->stage());
    graph->setStage(std::max(graph->stage(), input->stage()));
  }
  std::cout << " a2 \n";
  auto new_node = block->appendNode(graph->createClone(n, env, /*non-recursive */ false));
  new_node->setStage(n->stage());
  graph->setStage(std::max(graph->stage(), n->stage()));
  std::cout << " a3 \n";
  for(size_t i = 0; i < n->outputs().size(); ++i) {
    auto oo = n->outputs()[i];
    auto no = new_node->outputs()[i];
    local_map[oo] = no;
    no->copyMetadata(oo);
    no->setStage(oo->stage());
  }
  std::cout << " a4 \n";
  for(auto output : n->outputs()) {
    block->registerOutput(env(output));
  }
  std::cout << " a5 \n";
  auto values = fmap(block->inputs(), [&](Value* v) {
    return v->node()->t(attr::value);
  });
  std::cout << " a6 \n";
  InterpreterState(Code(graph)).runOneStage(values);
  std::cout << " a7 \n";
  std::cout << "Graph in prop: " << graph;
  return graph->outputs();
}

void propagateNode(Node *n) {
  // std::cout << "0\n";
  // std::cout << "propagating node: " << n->t(attr::value);
  // std::cout << "1\n";
  // for (auto n: n->inputs()) {
  //   std::cout << "node input: " << n->uniqueName() << "\n";
  //   std::cout << n->node()->t(attr::value);
  // }
  // std::cout << "\n";
  // std::cout << "end input: \n";
  auto outputs = runNode(n);
  for (auto n: outputs) {
    std::cout << "node output result: \n";
    if (n) std::cout << n->uniqueName() << "\n";
    if (n) std::cout << n->node()->t(attr::value);
  }
  for(size_t i = 0; i < n->outputs().size(); ++i) {
    // n->outputs()[i]->replaceAllUsesWith(outputs[i]);
  }
}

void ConstantPropagation(Node* n, bool recurse) {
  // std::cout << "0\n";
  bool propagate = (n->inputs().size() > 0) && std::all_of(n->inputs().begin(), n->inputs().end(), [&](Value* v) {
    return v->node()->kind() == prim::Constant;
  }) && n->kind() != prim::Print;
  std::cout << "propagating node prop: " << propagate << "\n";// << n->t(attr::value);
  // std::cout << "inputs: ";
  for (auto n: n->inputs()) {
    std::cout << "node input: " << n->uniqueName() << "\n"; //" " << n->node()->t(attr::value) <<
    // std::cout << n->node()->t(attr::value);
  }
  std::cout << "outputs: \n";
  for (auto n: n->outputs()) {
    std::cout << "node output: " << n->uniqueName() << "\n";
  }
  std::cout << " end outputs\n";
  if (propagate) {
    propagateNode(n);
  } 

  // bool propagate = std::all_of(n->inputs().begin(), n->inputs().end(), [&](Value* v) {
  //   return v->node()->kind() == prim::Constant;
  // }) && n->kind() != prim::Print;
  // if (propagate) {
  //   propagateNode(n);
  // } else {
  //   std::cout << "not propagating";
  //   for (auto n: n->inputs()) {
  //     std::cout << "node input: " << n->uniqueName() << "\n";
  //     std::cout << n->node()->t(attr::value);
  //   }
  // }
  if (recurse) {
    for (Block * block : n->blocks())
      ConstantPropagation(block, recurse);
  }
  // if (propagate) {
    // n->destroy();
  // }
}

void ConstantPropagation(Block* block, bool recurse) {
  std::cout << "visiting param node\n";
  ConstantPropagation(block->param_node(), recurse);
  for (auto n: block->nodes()) {
    std::cout << "visiting block node\n";
    ConstantPropagation(n, recurse);
  }
  std::cout << "visiting return node\n";
  ConstantPropagation(block->return_node(), recurse);
}

void ConstantPropagation(std::shared_ptr<Graph>& graph) {
  std::cout << "Hello\n";
  // std::cout << graph;
  // return;
  ConstantPropagation(graph->block(), true);
  // EliminateDeadCode(graph);

}

}}
